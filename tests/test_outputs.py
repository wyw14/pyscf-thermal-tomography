"""Behavior-level checks tied to the frozen public scientific contract."""

import copy
import hashlib
import json
import numpy as np
import pytest
from scipy import linalg
from candidate import call
from codec import encode
from independent import (
    build_case,
    exhaustive,
    arguments,
    finite_response,
    explicit_ghf,
    log_odds,
    coordinates,
    basis,
    native_thermodynamics,
)


class DeferredCandidateResult:
    """Run/cache the scientific call in the test body, so an API failure is a call failure."""

    def __init__(self, case):
        self.case = case
        self.value = None
        self.error = None

    def __getitem__(self, key):
        if self.value is None and self.error is None:
            args, kwargs = arguments(self.case)
            try:
                self.value = call("reconstruct", *args, **kwargs)
            except Exception as error:
                self.error = error
        if self.error is not None:
            raise self.error
        return self.value[key]


@pytest.fixture(scope="module")
def path_case():
    case = build_case()
    expected = exhaustive(case)
    return case, expected, DeferredCandidateResult(case)


def test_native_correspondence_density(path_case):
    """ "一个可行对应必须给出物理密度，并同时符合全部标记观测、固定约束及给定熵。" Molecular AO densities and recovered label directions."""
    case, expected, actual = path_case
    assert [tuple(v["mapping"]) for v in actual["frames"]] == list(
        expected["best"]["mapping_key"]
    )
    np.testing.assert_allclose(
        np.asarray([v["density"] for v in actual["frames"]]),
        np.asarray([v["density"] for v in expected["best"]["path"]]),
        atol=5e-7,
        rtol=3e-6,
        err_msg="frames[].density in the native nonorthogonal AO metric",
    )


def test_complete_physical_candidate_counts(path_case):
    """ "`candidate_counts[j]` 是第 j 条记录所有可行排列的数量，按排列计数而非按数值近似相同的密度合并。" Count physical ambiguities before path selection."""
    case, expected, actual = path_case
    assert expected["candidate_counts"] == [4, 4, 4, 4], (
        "Independent fixture must exercise four physical correspondences"
    )
    assert actual["candidate_counts"] == expected["candidate_counts"], (
        "candidate_counts must include every physical correspondence"
    )
    assert all(type(x) is int for x in actual["candidate_counts"])


@pytest.mark.parametrize("scenario", ["small_interior", "complex_native_corner"])
def test_global_minimum_vs_independent_enumeration(path_case, scenario):
    """ "所有帧共同参与最小化；追加观测后，既有帧的最优对应和参数代表元可以改变。" All 256 or 4096 native correspondence paths are independently optimized."""
    if scenario == "small_interior":
        case, expected, actual = path_case
        expected_parameters = expected["best"]["parameters"]
        expected_objective = expected["best"]["objective"]
        enumerated = expected["enumerated"]
        gap = expected["gap"]
    else:
        from global_corner import certified_corner

        case, certificate = certified_corner()
        args, kwargs = arguments(case)
        actual = call("reconstruct", *args, **kwargs)
        expected_parameters = certificate["parameters"]
        expected_objective = certificate["objective"]
        enumerated = certificate["enumerated"]
        gap = certificate["gap"]
    parameters = np.asarray(
        [np.r_[v["coefficients"], v["temperature"]] for v in actual["frames"]]
    )
    assert (
        enumerated == {"small_interior": 256, "complex_native_corner": 4096}[scenario]
        and gap > 1e-5
    ), "Independent fixture must completely certify a separated global correspondence"
    np.testing.assert_allclose(
        parameters,
        expected_parameters,
        atol=5e-7,
        rtol=3e-6,
        err_msg="jointly optimal complete native path parameters",
    )
    np.testing.assert_allclose(
        actual["objective"],
        expected_objective,
        atol=5e-7,
        rtol=3e-6,
        err_msg="global minimum J",
    )


def test_native_energy_and_constraint_duals(path_case):
    """ "约束乘子 lambda 的符号约定由变分拉格朗日量 F-sum_a lambda_a(Tr(A_a D)-c_a) 决定。" Explicit ERI energies and AO stationarity use the stated dual sign."""
    case, expected, actual = path_case
    row = actual["frames"][2]
    mode = expected["best"]["path"][2]
    dm = mode["density"]
    w = np.asarray(row["coefficients"])
    t = row["temperature"]
    potential = explicit_ghf(case["mol"], dm)
    energy = float(
        case["mf"].energy_nuc()
        + np.trace(
            (case["mf"].get_hcore() + np.einsum("a,aij->ij", w, case["generators"]))
            @ dm
        ).real
        + 0.5 * case["coupling"] * np.trace(dm @ potential).real
    )
    stationary = (
        case["mf"].get_hcore()
        + case["coupling"] * potential
        + np.einsum("a,aij->ij", w, case["generators"])
        + t * mode["thermal"]
    )
    represented = np.einsum(
        "a,aij->ij", row["constraint_multipliers"], case["constraints"]
    )
    np.testing.assert_allclose(
        row["energy"],
        energy,
        atol=5e-7,
        rtol=3e-6,
        err_msg="native electronic plus nuclear energy",
    )
    np.testing.assert_allclose(
        row["free_energy"],
        native_thermodynamics(case, row)["free_energy"],
        atol=5e-7,
        rtol=3e-6,
        err_msg="F=E-Ts with independent density-derived entropy and energy",
    )
    np.testing.assert_allclose(
        stationary,
        represented,
        atol=5e-7,
        rtol=3e-6,
        err_msg="AO variational stationarity and constraint multiplier sign",
    )


def thermodynamic_frames(case, complete, path_mode):
    """Prepare whole, prefix and resumed calls without deciding test outcomes."""
    if path_mode == "whole":
        return complete["frames"]
    args, kwargs = arguments(case, records=case["records"][:2])
    prefix = call("reconstruct", *args, **kwargs)
    if path_mode == "prefix":
        return prefix["frames"]
    if path_mode != "resumed":
        raise ValueError("Unknown thermodynamic path preparation")
    checkpoint = json.loads(json.dumps(prefix["checkpoint"], allow_nan=False))
    args, kwargs = arguments(
        case, records=case["records"][2:], checkpoint=checkpoint
    )
    return call("reconstruct", *args, **kwargs)["frames"]


@pytest.mark.parametrize("path_mode", ["whole", "prefix", "resumed"])
@pytest.mark.parametrize("observable", ["entropy", "free_energy"])
def test_native_thermodynamic_outputs(path_case, path_mode, observable):
    """"`entropy`、`energy`、`free_energy`：上面定义的实数。"

    Each returned frame, including revised checkpoint history, uses the native
    overlap metric's Fermi entropy and an independent electronic energy.
    """
    case, expected, complete = path_case
    frames = thermodynamic_frames(case, complete, path_mode)
    assert len(frames) == {"whole": 4, "prefix": 2, "resumed": 4}[path_mode], (
        "The complete nonempty requested prefix is returned"
    )
    expected_values = [native_thermodynamics(case, frame)[observable] for frame in frames]
    actual_values = [frame[observable] for frame in frames]
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        atol=5e-7,
        rtol=3e-6,
        err_msg=f"{path_mode} frames[].{observable} from native density",
    )


@pytest.mark.parametrize("native_state", ["unrun", "converged"])
def test_native_state_preserved_on_success(native_state):
    """"不得修改 mf、算符、输入记录或传入的 checkpoint。"

    Successful reconstruction preserves the actual supplied GHF orbital
    coefficients, occupations, orbital energies and convergence flag.
    """
    case = build_case("complex", count=1, queries=0)
    expected = exhaustive(case)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, _native_state=native_state, **kwargs)
    np.testing.assert_allclose(
        actual["frames"][0]["density"],
        expected["best"]["path"][0]["density"],
        atol=5e-7,
        rtol=3e-6,
        err_msg=f"{native_state} GHF supports the same physical reconstruction",
    )


@pytest.mark.parametrize("native_state", ["unrun", "converged"])
def test_native_state_preserved_on_rejection(native_state):
    """"不得修改 mf、算符、输入记录或传入的 checkpoint。"

    A rejected entropy record must preserve both unrun and converged native
    scientific state; the worker checks after catching the candidate exception.
    """
    case = build_case("complex", count=1, queries=0)
    bad_records = copy.deepcopy(case["records"])
    bad_records[0]["entropy"] = -10
    args, kwargs = arguments(case, records=bad_records)
    with pytest.raises(ValueError):
        call("reconstruct", *args, _native_state=native_state, **kwargs)


def test_noncollinear_complex_exchange():
    """ "使用完整的复数自旋密度与 GHF 库仑/交换作用；不假设自旋共线、矩阵为实数或 AO 正交。" Off-diagonal spin exchange affects the inferred fields."""
    case = build_case("complex", count=2, queries=0)
    expected = exhaustive(case)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, **kwargs)
    dm = expected["best"]["path"][0]["density"]
    native = case["mf"].get_veff(case["mol"], dm, hermi=1)
    assert np.max(np.abs(dm[:2, 2:])) > 1e-3, (
        "Fixture must contain noncollinear spin coherence"
    )
    np.testing.assert_allclose(
        explicit_ghf(case["mol"], dm),
        native,
        atol=2e-10,
        rtol=2e-10,
        err_msg="independent four-index Coulomb/exchange check",
    )
    np.testing.assert_allclose(
        np.asarray(
            [np.r_[r["coefficients"], r["temperature"]] for r in actual["frames"]]
        ),
        expected["best"]["parameters"],
        atol=5e-7,
        rtol=3e-6,
        err_msg="complex spinor self-consistent field inference",
    )


def test_complete_joint_gauge():
    """ "它描述观测约束下的完整局部参数族，不是全路径正则化后的 Hessian 逆。" Redundant generators, the chemical-potential gauge and thermal gauge coexist."""
    case = build_case("gauge", count=1, queries=0)
    expected = exhaustive(case)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, **kwargs)
    projector = np.asarray(actual["frames"][0]["joint_gauge_projector"])
    reference = expected["best"]["path"][0]["projector"]
    assert abs(np.trace(reference) - 3) < 1e-9, (
        "Fixture must exercise three independent parameter gauges"
    )
    np.testing.assert_allclose(
        projector,
        reference,
        atol=5e-7,
        rtol=3e-6,
        err_msg="complete joint observable-null projector",
    )
    np.testing.assert_allclose(
        projector @ projector,
        projector,
        atol=5e-7,
        rtol=3e-6,
        err_msg="idempotent gauge projection",
    )
    np.testing.assert_allclose(
        np.r_[actual["frames"][0]["coefficients"], actual["frames"][0]["temperature"]],
        expected["best"]["parameters"][0],
        atol=5e-7,
        rtol=3e-6,
        err_msg="anchor-compatible representative in the redundant family",
    )


def test_active_temperature_response():
    """ "`temperature_bound_multiplier`：有符号的温度边界乘子，即 J 对该帧 T 的偏导；内部为0，下界非负，上界非正。" The active physical temperature bound changes the constrained tangent."""
    case = build_case("active", count=1, queries=1)
    expected = exhaustive(case)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, **kwargs)
    finite = finite_response(case, 0)
    assert abs(expected["best"]["parameters"][0, -1] - case["interval"][1]) < 1e-7, (
        "Fixture must reach a strictly active upper bound"
    )
    assert expected["best"]["temperature_derivative"][0] < -1e-4, (
        "Bound must have strict complementarity"
    )
    np.testing.assert_allclose(
        actual["frames"][0]["temperature_bound_multiplier"],
        expected["best"]["temperature_derivative"][0],
        atol=5e-7,
        rtol=3e-6,
        err_msg="signed boundary multiplier without a hidden 1/2",
    )
    np.testing.assert_allclose(
        actual["frames"][0]["parameter_response"][0],
        finite["parameters"][0],
        atol=3e-6,
        rtol=1e-5,
        err_msg="active-bound constrained parameter response",
    )


def test_nonlocal_parameter_response(path_case):
    """ "响应是联合最优解对上述输入方向的一阶导数，必须考虑所有帧及给定锚点的变化。" Future measurements move earlier gauge representatives."""
    case, expected, actual = path_case
    finite0 = finite_response(case, 0)
    finite1 = finite_response(case, 1)
    assert np.max(np.abs(finite0["parameters"][:-1])) > 1e-5, (
        "Fixture must have a nonlocal future-to-prefix response"
    )
    np.testing.assert_allclose(
        np.asarray([v["parameter_response"][0] for v in actual["frames"]]),
        finite0["parameters"],
        atol=3e-6,
        rtol=1e-5,
        err_msg="future-to-prefix parameter derivative",
    )
    np.testing.assert_allclose(
        np.asarray([v["parameter_response"][1] for v in actual["frames"]]),
        finite1["parameters"],
        atol=3e-6,
        rtol=1e-5,
        err_msg="joint observation and anchor parameter derivative",
    )
    np.testing.assert_allclose(
        actual["objective_response"],
        [finite0["objective"], finite1["objective"]],
        atol=3e-6,
        rtol=1e-5,
        err_msg="global objective directional derivatives",
    )


def test_physical_density_response(path_case):
    """ "未来帧的查询可以改变过去帧的参数响应，而过去帧的物理密度仍由该帧的完整观测决定。" Separate observable density response from representative response."""
    case, expected, actual = path_case
    finite = finite_response(case, 0)
    np.testing.assert_allclose(
        np.asarray([v["density_response"][0] for v in actual["frames"]]),
        finite["density"],
        atol=3e-6,
        rtol=1e-5,
        err_msg="complex native AO density response",
    )
    np.testing.assert_allclose(
        np.asarray([v["density_response"][0] for v in actual["frames"][:-1]]),
        0,
        atol=3e-6,
        rtol=0,
        err_msg="future query cannot alter fixed past tomography",
    )


def test_checkpoint_revises_prefix(path_case):
    """ "相同上下文的整段调用与任意合法分批续接，必须返回相同的完整前缀最优结果和响应；检查点字节无需与整段结果相同。" A prefix optimum alone is insufficient continuation state."""
    case, expected, actual = path_case
    args, kwargs = arguments(case, records=case["records"][:2])
    prefix = call("reconstruct", *args, **kwargs)
    checkpoint = json.loads(json.dumps(prefix["checkpoint"], allow_nan=False))
    args, kwargs = arguments(case, records=case["records"][2:], checkpoint=checkpoint)
    resumed = call("reconstruct", *args, **kwargs)
    old = np.asarray(
        [np.r_[x["coefficients"], x["temperature"]] for x in prefix["frames"]]
    )
    new = np.asarray(
        [np.r_[x["coefficients"], x["temperature"]] for x in actual["frames"][:2]]
    )
    assert np.max(np.abs(old - new)) > 1e-4, (
        "Fixture must require revising an earlier representative"
    )
    assert (
        len(resumed["frames"]) == 4
        and resumed["candidate_counts"] == expected["candidate_counts"]
    )
    np.testing.assert_allclose(
        np.asarray(
            [np.r_[v["coefficients"], v["temperature"]] for v in resumed["frames"]]
        ),
        expected["best"]["parameters"],
        atol=5e-7,
        rtol=3e-6,
        err_msg="continued complete global path after prefix revision",
    )


def test_resumed_response_is_global(path_case):
    """ "payload 内部表示自定，必须足以恢复相同完整前缀及其查询。" JSON round trip must preserve query dependence across the old/new boundary."""
    case, expected, actual = path_case
    args, kwargs = arguments(case, records=case["records"][:1])
    first = call("reconstruct", *args, **kwargs)
    args, kwargs = arguments(
        case,
        records=case["records"][1:3],
        checkpoint=json.loads(json.dumps(first["checkpoint"])),
    )
    middle = call("reconstruct", *args, **kwargs)
    args, kwargs = arguments(
        case,
        records=case["records"][3:],
        checkpoint=json.loads(json.dumps(middle["checkpoint"])),
    )
    resumed = call("reconstruct", *args, **kwargs)
    np.testing.assert_allclose(
        np.asarray([v["parameter_response"] for v in resumed["frames"]]),
        np.asarray([v["parameter_response"] for v in actual["frames"]]),
        atol=3e-6,
        rtol=1e-5,
        err_msg="all revised prefix sensitivities after two process-independent resumes",
    )
    np.testing.assert_allclose(
        resumed["objective_response"],
        actual["objective_response"],
        atol=3e-6,
        rtol=1e-5,
        err_msg="resumed global objective sensitivity",
    )


def test_thirty_two_frame_path():
    """ "整个会话最多32条记录，最多4个未标记通道。" A long ambiguous path has the independently certified zero-jump global solution."""
    case = build_case("long", count=32, queries=0)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, **kwargs)
    params = np.asarray(
        [np.r_[row["coefficients"], row["temperature"]] for row in actual["frames"]]
    )
    assert len(actual["frames"]) == 32 and actual["candidate_counts"] == [4] * 32
    np.testing.assert_allclose(
        params,
        np.repeat(case["anchor"][None], 32, axis=0),
        atol=5e-7,
        rtol=3e-6,
        err_msg="all 32 globally constant representatives with independent J>=0 certificate",
    )
    np.testing.assert_allclose(
        actual["objective"],
        0,
        atol=5e-7,
        rtol=0,
        err_msg="zero-jump global lower bound is attained",
    )


@pytest.mark.parametrize("kind,size", [("two", 2), ("six", 6), ("eight", 8)])
def test_supported_molecular_sizes(kind, size):
    """ "支持域：2<=n<=8，1<=r<=4，m<=n*n+4，0<=q<=4，整个会话最多32条记录，最多4个未标记通道。" Native open-shell atom and three- and four-center nonorthogonal molecules."""
    case = build_case(kind, count=1, queries=0)
    expected = exhaustive(case)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, **kwargs)
    assert np.asarray(actual["frames"][0]["density"]).shape == (size, size)
    np.testing.assert_allclose(
        actual["frames"][0]["density"],
        expected["best"]["path"][0]["density"],
        atol=5e-7,
        rtol=3e-6,
        err_msg="native supported-size AO reconstruction",
    )
    np.testing.assert_allclose(
        np.r_[actual["frames"][0]["coefficients"], actual["frames"][0]["temperature"]],
        expected["best"]["parameters"][0],
        atol=5e-7,
        rtol=3e-6,
        err_msg="supported-size canonical parameter reconstruction",
    )


def test_empty_and_zero_query_shapes():
    """ "没有记录时可建立或复读检查点，结果为空前缀 J=0，方向形状保留。" Output keys and zero-length scientific axes remain explicit."""
    case = build_case(count=1, queries=2)
    args, kwargs = arguments(case, records=[])
    empty = call("reconstruct", *args, **kwargs)
    assert set(empty) == {
        "frames",
        "objective",
        "objective_response",
        "candidate_counts",
        "checkpoint",
    }
    assert (
        empty["frames"] == []
        and empty["candidate_counts"] == []
        and empty["objective"] == 0
    )
    assert np.asarray(empty["objective_response"]).shape == (2,)
    zero = build_case("complex", count=1, queries=0)
    args, kwargs = arguments(zero)
    answer = call("reconstruct", *args, **kwargs)
    row = answer["frames"][0]
    assert set(row) == {
        "mapping",
        "density",
        "coefficients",
        "temperature",
        "constraint_multipliers",
        "joint_gauge_projector",
        "entropy",
        "energy",
        "free_energy",
        "parameter_response",
        "density_response",
        "temperature_bound_multiplier",
    }
    assert np.asarray(row["parameter_response"]).shape == (
        0,
        len(zero["generators"]) + 1,
    )
    assert np.asarray(row["density_response"]).shape == (0, 4, 4)


def test_rejection_preserves_prior(path_case):
    """ "批次后部拒收时，传入的 checkpoint 和此前返回的对象保持原值。" Invalid later tomography and incompatible entropy directions are rejected atomically."""
    case, expected, actual = path_case
    args, kwargs = arguments(case, records=case["records"][:1])
    first = call("reconstruct", *args, **kwargs)
    checkpoint = json.loads(json.dumps(first["checkpoint"]))
    wire = json.dumps(checkpoint, sort_keys=True)
    bad = copy.deepcopy(case["records"][1:])
    bad[-1]["entropy"] = -10
    args, kwargs = arguments(case, records=bad, checkpoint=checkpoint)
    with pytest.raises(ValueError):
        call("reconstruct", *args, **kwargs)
    bad = copy.deepcopy(case["records"][1:])
    bad[-1]["directions"][0, -1] += 0.01
    args, kwargs = arguments(case, records=bad, checkpoint=checkpoint)
    with pytest.raises(ValueError):
        call("reconstruct", *args, **kwargs)
    assert json.dumps(checkpoint, sort_keys=True) == wire
    args, kwargs = arguments(case, records=case["records"][1:], checkpoint=checkpoint)
    recovered = call("reconstruct", *args, **kwargs)
    np.testing.assert_allclose(
        recovered["objective"],
        actual["objective"],
        atol=5e-7,
        rtol=3e-6,
        err_msg="valid recovery from the unchanged prior after rejected batches",
    )
    args, kwargs = arguments(case)
    args[1] = np.asarray(args[1], complex).copy()
    args[1][0, 0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError):
        call("reconstruct", *args, **kwargs)


@pytest.mark.parametrize("payload_kind", ["null", "array", "number"])
def test_scientific_context_and_checksum(path_case, payload_kind):
    """ "科学上下文改变时必须拒收旧检查点；上下文包含原生 S、h0、两电子积分、Enuc、算符、固定目标、未标记索引、coupling 和温度区间。" The state cannot silently migrate to another Hamiltonian."""
    case, expected, actual = path_case
    cp = json.loads(json.dumps(actual["checkpoint"]))
    assert set(cp) == {"schema_version", "context", "payload", "sha256"}
    core = {key: cp[key] for key in ("schema_version", "context", "payload")}
    digest = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert (
        cp["schema_version"] == 1
        and cp["sha256"] == digest
        and type(cp["context"]) is str
        and len(cp["context"]) > 0
    )
    corrupt = copy.deepcopy(cp)
    corrupt["payload"]["tampered"] = 1
    args, kwargs = arguments(case, records=[], checkpoint=corrupt)
    with pytest.raises(ValueError):
        call("reconstruct", *args, **kwargs)
    args, kwargs = arguments(case, records=[], checkpoint=cp)
    kwargs["coupling"] = 0.7
    with pytest.raises(ValueError):
        call("reconstruct", *args, **kwargs)
    malformed = copy.deepcopy(cp)
    malformed["payload"] = {"null": None, "array": [], "number": 17}[payload_kind]
    core = {key: malformed[key] for key in ("schema_version", "context", "payload")}
    malformed["sha256"] = hashlib.sha256(
        json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    args, kwargs = arguments(case, records=[], checkpoint=malformed)
    with pytest.raises(ValueError):
        call("reconstruct", *args, **kwargs)


def test_extra_fixed_constraint():
    """ "`constraints` 给出 r 个固定 Hermitian 算符 A_a，`targets` 给出 Tr(A_a D)=c_a；第一个 A 为 S，第一个 c 为 mf.mol.nelectron。" A second moment changes the feasible density and multiplier space."""
    case = build_case("constraint", count=2, queries=1)
    expected = exhaustive(case)
    args, kwargs = arguments(case)
    actual = call("reconstruct", *args, **kwargs)
    row = actual["frames"][1]
    moments = np.einsum(
        "aij,ji->a", case["constraints"], np.asarray(row["density"])
    ).real
    assert np.asarray(row["constraint_multipliers"]).shape == (2,)
    np.testing.assert_allclose(
        moments,
        case["targets"],
        atol=5e-7,
        rtol=3e-6,
        err_msg="all independent fixed Hermitian constraints",
    )
    np.testing.assert_allclose(
        np.r_[row["coefficients"], row["temperature"]],
        expected["best"]["parameters"][1],
        atol=5e-7,
        rtol=3e-6,
        err_msg="canonical representative in the constrained chemical family",
    )

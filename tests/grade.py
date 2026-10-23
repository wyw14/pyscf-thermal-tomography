"""Trusted lifecycle/integrity gate, with weighted reward only after full execution."""
from pathlib import Path
import hashlib, json, os, sys, uuid, xml.etree.ElementTree as ET
sys.path.insert(0,'/tests')
from score import aggregate
ROOT=Path('/logs/verifier');TESTS=Path('/tests')


def write_reward(value):
    ROOT.mkdir(parents=True,exist_ok=True)
    temporary=ROOT/'reward.tmp';temporary.write_text(str(value)+'\n',encoding='utf-8')
    os.replace(temporary,ROOT/'reward.txt')


def file_hash(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def integrity(contract):
    for name,expected in contract['test_files_sha256'].items():
        actual=(TESTS/name).resolve()
        if not actual.is_relative_to(TESTS) or file_hash(actual)!=expected:
            raise RuntimeError('Trusted verifier changed: '+name)


def main():
    action=sys.argv[1]
    if action in ('initialize','abort'):write_reward(0)
    if action=='abort':return
    contract=json.loads((TESTS/'test-contract.json').read_text());integrity(contract)
    if action=='initialize':
        for name in ('junit.xml','execution.json','grade-decision.json'):
            (ROOT/name).unlink(missing_ok=True)
        (ROOT/'run-identity.json').write_text(json.dumps({'run_id':uuid.uuid4().hex}))
        return
    if action=='check-wheels':
        for name,expected in contract['wheel_files_sha256'].items():
            if file_hash(TESTS/'wheels'/name)!=expected:raise RuntimeError('Verifier wheel changed: '+name)
        return
    if action!='finish':raise ValueError(action)
    write_reward(0)
    run=json.loads((ROOT/'run-identity.json').read_text())
    execution=json.loads((ROOT/'execution.json').read_text());expected=contract['nodeids'];reasons=[]
    if sys.argv[2] not in ('0','1'):reasons.append('Pytest infrastructure or interrupted exit '+sys.argv[2])
    if not expected or sorted(execution.get('collected',[]))!=sorted(expected) or len(set(expected))!=len(expected):reasons.append('Incomplete or changed test collection')
    if execution.get('run_id')!=run['run_id']:reasons.append('Execution identity mismatch')
    if execution.get('session_finished') is not True or execution.get('exitstatus') not in (0,1):reasons.append('Suite did not complete normally')
    for node in expected:
        phases=execution.get('reports',{}).get(node,{})
        if set(phases)!={'setup','call','teardown'}:reasons.append('Incomplete lifecycle '+node);continue
        if phases['setup']['outcome']!='passed' or phases['teardown']['outcome']!='passed' or phases['call']['outcome'] not in ('passed','failed'):reasons.append('Skipped or invalid lifecycle '+node)
    cases=list(ET.parse(ROOT/'junit.xml').iter('testcase'))
    if len(cases)!=len(expected) or any(case.find('skipped') is not None or case.find('error') is not None for case in cases):reasons.append('JUnit completion mismatch')
    breakdown=[];reward=0.0
    if not reasons:
        profile=json.loads((TESTS/'scoring.json').read_text())
        reward,breakdown=aggregate(expected,execution['reports'],profile)
        failed=sum(execution['reports'][n]['call']['outcome']=='failed' for n in expected)
        xml_failed=sum(c.find('failure') is not None for c in cases)
        if failed!=xml_failed or (failed==0)!=(execution['exitstatus']==0):reasons.append('Lifecycle and JUnit outcomes disagree');reward=0.0
    decision={'run_id':run['run_id'],'expected_cases':len(expected),'actual_cases':len(cases),'reasons':reasons,'weighted_behaviors':breakdown,'reward':reward}
    (ROOT/'grade-decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    write_reward(reward)
    if reasons:raise RuntimeError('; '.join(reasons))


if __name__=='__main__':
    try:main()
    except BaseException:
        write_reward(0)
        raise

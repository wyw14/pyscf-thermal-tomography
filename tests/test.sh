#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
chmod 700 /logs/verifier /tests
if [ -d /solution ]; then chmod 700 /solution; fi
reset_reward() {
    printf '0\n' > /logs/verifier/reward.initializing
    mv -f /logs/verifier/reward.initializing /logs/verifier/reward.txt
}
reset_reward
fail_closed() {
    status=$?
    if [ "$status" -ne 0 ]; then reset_reward; fi
}
trap fail_closed EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
python -I /tests/grade.py initialize
python -I /tests/grade.py check-wheels
python -m pip install --no-index --find-links=/tests/wheels --require-hashes -r /tests/requirements.lock.txt
chmod 0444 /tests/*.py /tests/*.ini /tests/*.json
python -m pip freeze > /logs/verifier/environment.txt
cd /tests
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tests \
  timeout --signal=TERM --kill-after=5s 840s \
  python -m pytest /tests/test_outputs.py -q -rA --tb=short -p no:cacheprovider \
  -p trusted_events --confcutdir=/tests --rootdir=/tests -c /tests/pytest.ini \
  --junitxml=/logs/verifier/junit.xml
status=$?
set -e
python -I /tests/grade.py finish "$status"

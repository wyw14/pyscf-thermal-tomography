#!/bin/bash
set -euo pipefail
PATH=/usr/local/bin:/usr/bin:/bin
export PATH
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTEST_ADDOPTS PYTEST_PLUGINS
mkdir -p /logs/verifier
reset_reward() {
    rm -f /logs/verifier/reward.json
    printf '0\n' > /logs/verifier/reward.initializing
    mv -f /logs/verifier/reward.initializing /logs/verifier/reward.txt
}
reset_reward
fail_closed() {
    status=$?
    if [ "$status" -ne 0 ]; then reset_reward; fi
}
trap fail_closed EXIT
chmod 700 /logs/verifier
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
/usr/local/bin/python -B -I /tests/grade.py initialize
/usr/local/bin/python -B -I /tests/grade.py check-wheels
/usr/local/bin/python -B -I -m pip install --no-index --find-links=/tests/wheels --require-hashes -r /tests/requirements.lock.txt
/usr/local/bin/python -B -I /tests/runtime_sandbox.py > /logs/verifier/isolation-preflight.log 2>&1
/usr/local/bin/python -B -I -m pip freeze > /logs/verifier/environment.txt
cd /tests
set +e
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tests \
  /usr/bin/timeout --signal=TERM --kill-after=5s 840s \
  /usr/local/bin/python -B -s -P -m pytest /tests/test_outputs.py -q -rA --tb=short -p no:cacheprovider \
  -p trusted_events --confcutdir=/tests --rootdir=/tests -c /tests/pytest.ini \
  --junitxml=/logs/verifier/junit.xml
status=$?
set -e
/usr/local/bin/python -B -I /tests/grade.py finish "$status"

from celery_app import celery_app

def test_test_task():
    result = celery_app.send_task('celery_app.test_task', args=(2, 3))
    assert result.get(timeout=10) == 5

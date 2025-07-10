from celery import shared_task


@shared_task(bind=True)
def process_video_task(self, url):
    import subprocess
    import json

    try:
        result = subprocess.run(
            ['python3', 'mediapipe_api/process_worker.py', url],
            capture_output=True, text=True, timeout=600
        )
        print("[Worker stdout]:", result.stdout)
        print("[Worker stderr]:", result.stderr)

        if result.returncode != 0:
            return {'error': result.stderr.strip() or "Script failed with non-zero exit."}

        return json.loads(result.stdout)  # This is where it fails
    except subprocess.TimeoutExpired:
        return {'error': 'Worker script timed out after 600s'}
    except json.JSONDecodeError:
        return {
            'error': 'Worker script did not return valid JSON',
            'raw_output': result.stdout.strip()
        }
import traceback
from app.main import app
from fastapi.testclient import TestClient

try:
    client = TestClient(app)
    response = client.post('/v1/score', json={
        'thumbnails': [{'id': 'test', 'url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='}],
        'title': 'Test Video',
        'niche': 'food'
    })
    print('Status:', response.status_code)
    if response.status_code == 200:
        print('Success\!')
    else:
        print('Error response:', response.text)
except Exception as e:
    print('Exception:', e)
    traceback.print_exc()

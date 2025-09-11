import base64
import os
from flask import Flask, request, jsonify
from google.cloud import aiplatform

# --- 설정 (수정할 필요 없음) ---
PROJECT_ID = "vision02"
LOCATION = "us-central1"
ENDPOINT_ID = os.environ.get("ENDPOINT_ID")
# ------------------------------------

app = Flask(__name__)

# Vertex AI 클라이언트 초기화 (Cloud Run 환경에서는 자동으로 처리되므로, 
# 만약 이 라인에서 오류 발생 시 제거해도 괜찮습니다.)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.route("/predict", methods=["POST"])
def predict():
    """Flutter 앱으로부터 이미지 데이터를 받아 예측을 수행합니다."""
    # 1. 앱에서 보낸 JSON 데이터 받기 (이 부분은 한번만 있어야 합니다)
    data = request.get_json()
    if not data or 'image_bytes' not in data:
        return jsonify({"error": "image_bytes (base64) is required"}), 400

    # try...except 블록은 함수 내부에 올바르게 들여쓰기 되어야 합니다.
    try:
        # 2. Base64로 인코딩된 이미지 데이터를 디코딩
        image_bytes = base64.b64decode(data['image_bytes'])
        
        # 3. Vertex AI Endpoint 객체 초기화
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        
        # 4. 예측 요청 데이터 구성
        encoded_content = base64.b64encode(image_bytes).decode("utf-8")
        
        instances = [
            {"content": encoded_content}
        ]
        
        parameters = {
            "confidenceThreshold": 0.5,
            "maxPredictions": 5
        }

        # 5. Vertex AI에 예측 요청 실행
        response = endpoint.predict(instances=instances, parameters=parameters)

        # 6. 예측 결과 파싱
        prediction_result = response.predictions[0]
        
        display_name = prediction_result.get('displayNames', [None])[0]
        confidence = prediction_result.get('confidences', [None])[0]

        if display_name is None or confidence is None:
            raise ValueError("Failed to parse prediction response. Check displayNames and confidences fields.")

        # 7. 최종 결과 반환
        return jsonify({
            'class': display_name,
            'confidence': confidence
        })
    
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
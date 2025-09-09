import base64
import os
from flask import Flask, request, jsonify
from google.cloud import aiplatform

# --- 설정 (본인의 환경에 맞게 수정) ---
PROJECT_ID =  "vision02" # 👈 본인의 GCP 프로젝트 ID
LOCATION = "us-central1"           # 👈 Vertex AI Endpoint를 배포한 리전
ENDPOINT_ID = "endpoint-id-here" # 👈 Vertex AI의 Endpoint ID
# ------------------------------------

app = Flask(__name__)

# Vertex AI 클라이언트 초기화
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.route("/predict", methods=["POST"])
def predict():
    """Flutter 앱으로부터 이미지 데이터를 받아 예측을 수행합니다."""
    # 1. 앱에서 보낸 JSON 데이터 받기
    data = request.get_json()
    if not data or 'image_bytes' not in data:
        return jsonify({"error": "image_bytes (base64) is required"}), 400

    try:
        # 2. Base64로 인코딩된 이미지 데이터를 디코딩
        image_bytes = base64.b64decode(data['image_bytes'])
        
        # 3. Vertex AI Endpoint에 예측 요청 보내기
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        
        # Vertex AI 온라인 예측 요청 형식에 맞게 인스턴스 구성
        # AutoML 이미지 객체 감지 모델의 경우 아래와 같은 형식을 따를 수 있습니다.
        instances = [
            {
                "content": base64.b64encode(image_bytes).decode("utf-8")
            }
        ]
        
        response = endpoint.predict(instances=instances)
        
        # 4. Vertex AI의 응답을 가공하여 Flutter 앱에 전달
        # (응답 구조는 모델에 따라 다를 수 있으므로, 실제 응답을 보고 수정 필요)
        prediction = response.predictions[0]
        display_name = prediction['displayNames'][0] # 예: 'ugly' 또는 'good'
        confidence = prediction['confidences'][0]     # 예: 0.98

        print(f"Prediction: {display_name} with confidence {confidence}")

        return jsonify({
            "class": display_name,
            "confidence": confidence
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


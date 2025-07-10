# api/services/postService/send_yolo.py
import requests

video_path = "D:\\Desktop\\abnormal_5.mp4"
response = requests.post("http://localhost:5000/detect-abnormal", json={"video_path": video_path})

if response.ok:
    data = response.json()
    print("✅ 检测完成")
    print("检测信息：", data["message"])
    print("处理后视频地址：", data["video_url"])
else:
    print("请求失败:", response.text)

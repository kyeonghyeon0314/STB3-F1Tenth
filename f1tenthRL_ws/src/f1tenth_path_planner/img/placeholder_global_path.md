# Global Path 이미지 플레이스홀더

실제 경로 계획 패키지 실행 후 스크린샷을 찍어서 다음 이미지들을 추가하세요:

1. **global_path.png**: 맵 상의 센터라인 기반 전역 경로
2. **local_path.png**: LiDAR 기반 지역 장애물 회피 경로  
3. **path_visualization.png**: RViz에서 global_path와 local_path가 함께 보이는 화면

이미지 추가 후 README.md에 다음과 같이 포함:

```markdown
## 📸 시각화 예시

<div align="center">
  <img src="img/global_path.png" alt="Global Path Example" width="400">
  <p><em>센터라인 기반 Global Path 예시</em></p>
</div>

<div align="center">
  <img src="img/local_path.png" alt="Local Path Example" width="400">
  <p><em>LiDAR 기반 Local Path 예시</em></p>
</div>

<div align="center">
  <img src="img/path_visualization.png" alt="Path Visualization" width="500">
  <p><em>RViz에서 Global Path와 Local Path 시각화</em></p>
</div>
```
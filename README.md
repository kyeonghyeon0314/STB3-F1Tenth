
# 도커 환경 구성
./scripts/docker.sh




# 실행 명령어
```
ros2 launch f1tenth f1tenth_checkpoint_system_launch.py model_path:=/home/STB3-F1Tenth/f1tenthRL_ws/src/models/sac-cnn-8th.zip speed_max:=5.0


python3 training_sac_cnn.py --debug --target-entropy -2.0 --save
```
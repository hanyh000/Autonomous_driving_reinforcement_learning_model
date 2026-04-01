# TurtleBot3 DQN Autonomous Navigation

> **"DQN 강화학습과 YOLO 객체 인식을 결합한 터틀봇 자율주행 시스템"**
> ROS2 기반 Gazebo 시뮬레이션에서 학습된 모델을 실제 TurtleBot3(Waffle Pi)에 배포하여 장애물 회피 및 목적지 주행을 성공시킨 프로젝트입니다.

![Category](https://img.shields.io/badge/Project-Robotics%20&%20AI-blue)
![Framework](https://img.shields.io/badge/Framework-ROS2%20Humble%20|%20TensorFlow-orange)
![Award](https://img.shields.io/badge/Award-Best%20Project%20Winner-gold)


---


## 1. 프로젝트 개요

* **배경**: 단순 센서 기반 주행의 한계를 넘어, 강화학습을 통해 복잡한 장애물 환경에서도 최적의 경로를 스스로 찾아내는 자율주행 에이전트 구현.
* **핵심 기술**: DQN(Deep Q-Network), ROS2 통신, YOLO 객체 인식, Sim-to-Real 전이.
* **기간**: 2026.01.06 ~ 2026.01.20 (2주)
* **성과**: **미래융합교육원 로봇 개발자 과정 심화 프로젝트 우수상** 수상.


---


## 2. DQN 모델 아키텍처

| 항목 | 상세 내용 |
| :--- | :--- |
| **Input (State)** | 26차원 (LiDAR 데이터 + 상대 목표 거리 및 각도) |
| **Output (Action)** | 7가지 선속도 및 각속도 조합 |
| **Hidden Layers** | Dense(512) -> Dense(256) -> Dense(128) (ReLU) |
| **Hyperparameters** | Learning Rate: 0.0007, Discount Factor: 0.99, Batch Size: 32 |


---


## 3. 프로젝트 구조

```text
.
├── dqn_custom_agent.py         # DQN 에이전트 (학습 루프 및 신경망 정의)
├── dqn_custom_environment.py   # ROS2 환경 인터페이스 (State 정의 및 보상 계산)
├── dqn_custom_gazebo.py        # 시뮬레이션 인터페이스 (Entity 관리)
├── dqn_custom_test.py          # 학습 모델 검증 및 실제 로봇 배포용 스크립트
└── saved_model/                # 최적의 가중치 파일 (.h5, .json)
```

---


## 4. 주요 코드 구현 (Reward Function)

로봇이 목적지에 도달하면 큰 보상을 주고, 충돌 시 패널티를 부여하여 최적의 경로를 학습하게 하는 핵심 알고리즘입니다.

```python
def get_reward(self, action, done, collision, arrive):
    # 1. 목적지 도달 시 큰 보상
    if arrive:
        reward = 1000.0
    # 2. 장애물 충돌 시 큰 패널티
    elif collision:
        reward = -500.0
    # 3. 목표 지점에 가까워질수록 보상 가중 (Distance-based)
    else:
        reward = self.get_distance_reward()

    return reward
'''


---


## 5. 주요 기술적 해결 (Troubleshooting)

### **a. Sim-to-Real Gap 극복**
* **문제**: 가상 환경(Gazebo)에서 완벽하게 주행하던 모델이 실제 환경의 센서 노이즈와 물리적 마찰로 인해 회전 각도가 틀어지는 문제 발생.
* **해결**: 실제 로봇의 센서 데이터를 시뮬레이션 학습 시 노이즈로 추가(Data Augmentation)하고, 보상 구조(Reward Shaping)에서 회전 반경에 대한 제약 조건을 강화하여 부드러운 주행 유도.

### **b. YOLO 객체 인식과 강화학습 노드 통합**
* **문제**: 객체 인식 노드와 DQN 제어 노드 간 통신 지연으로 인한 실시간성 저하.
* **해결**: ROS2의 **Lifecycle Node** 개념을 응용하여 데이터 동기화를 최적화하고, 이미지 처리 부하를 줄이기 위해 인식 주기를 조절하여 제어 주기를 확보함.

---


## 6. 기여도 및 성과

* **기여도**: **State/Reward 구조 설계 (본인)**, 환경 인터페이스 노드 구현 및 전체 시스템 통합.

* **주요 성과**:
    * 시뮬레이션(Gazebo) 학습 모델을 실제 로봇에 성공적으로 이식 (**Sim-to-Real**).
    * YOLO 객체 인식을 통한 화살표 방향 기반 경로 추종 기능 구현.
    * **우수상 수상** (미래융합교육원 로봇 개발자 과정 심화 프로젝트).

---


## 7. 결과 및 시연

* **시연 영상**: [객체 인식 적용 전/후 비교 영상](링크를_입력하세요)
* **결과**: 우수상 수상

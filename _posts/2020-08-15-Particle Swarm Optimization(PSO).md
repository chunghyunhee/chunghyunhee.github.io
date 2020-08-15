

- PSO는 새나 물고기 무리 등의 사회적 행동 양식에 대한 규칙성을 증명하는 것에 착안하여 개발한 알고리즘이다. 
- 나중에 단순화 되었으며, 최적화 문제를 해결하는데 사용되어짐

# refer
1.Adaptive Nulling algorithm for null synthesus on the moving Jammer Environment
- PSO알고리즘은 자기학습을 통해 최적의 해를 찾는 최적화 알고리즘이다. 
- 알고리즘이 매우 간단하고 구현이 용이

2. http://astl.snu.ac.kr/Research/mdo04.asp?tp1=025
- PSO는 군집 개체를 모방하였다는 점에서 GA와 유사함
- PSO는 GA를 사용했을 때와 거의 유사한 결과를 가진다. 
- 알고리즘 특성상 설계변수가 복잡한 경우에 수렴 시간이 단축
- 복잡한 연산에 대해 전역적인 최적화 가능

3. Kennedy, James. "Particle swarm optimization." Encyclopedia of machine learning. Springer, Boston, MA
- 저자가 개발한 PSO 방법은 매우 간단한 개념으로 구현이 가능함
- 원시적인 수학 연산만이 필요하며 메모리 요구사항과 속도 측면에서 계산 비용이 저렴하다

# PSO의 개념
- PSO는 swarm이라고 불리는 particle의 bunch를 사용
- 이 particle들은 search-space를 돌아다니면서 탐색한다. 
- 이 particle들은 조건에 의해 안내되는 방향으로 이동한다 
	1. 관성(inertia) : particle 자신의 이전 속도
	2. 인지력(cognitive force) : 각 particle의 best known position으로부터의 거리
	3. 사회력(social force) : swarm best known position과의 거리 

![image](https://user-images.githubusercontent.com/49298791/90208363-a3b52200-de23-11ea-9a06-56f71f55bbf8.png)

- 각 particle은 자신이 이전에 이동한 위치를 알고있고, 자신이 지났던 위치중에 가장 최적의 값을 가지는 위치를 기억하며, swarm내에서 최적의 값을 가지는 위치를 공유한다. 
- 위의 세가지 방향을 고려하여 새로 나아갈 position을 탐색한다. 
![image](https://user-images.githubusercontent.com/49298791/90208172-212c6280-de23-11ea-8c19-f121eab080c5.png)



# flow chart
![image](https://user-images.githubusercontent.com/49298791/90304817-74bbb080-def6-11ea-98d0-95a53d94b98f.png)


# pseudo code
![image](https://user-images.githubusercontent.com/49298791/90214370-34472e80-de33-11ea-8897-b66b1036ed6a.png)

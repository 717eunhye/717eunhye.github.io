---
layout: post
title: Python-Line chart 사용법
---
본 포스팅은 line chart를 활용하여 주로 사용했던 방법과 옵션을 정리하였다. 

line chart는 아래 두 줄 만으로도 차트가 그려진다. 

<script src="https://gist.github.com/717eunhye/1ea4041339f82786d81e0af0deea380e.js"></script>

![image1](https://user-images.githubusercontent.com/50131912/163088616-26bdb33e-c963-4ae2-8f0b-2fb8bb8c3c6d.png)

## **1. 단일차트**

<script src="https://gist.github.com/717eunhye/ff1b64004a7c2182c9de0f962487466f.js"></script>

![image2](https://user-images.githubusercontent.com/50131912/163089172-ac86716a-af60-4645-828c-aa47c02463c3.png)

```
 1) 차트 크기 :  plt.figure(figsize=(max, min))  
 2) 범례 추가 : plt.plot(x, y, label = 'y') 안에 y값의 범례명을 적어 준 후 plt.legend() 함수 사용  
 3) 축 라벨링 : x축의 경우 plt.xlabel("x축명") , plt.ylabel("y축명")  
 4) 차트 제목 :  plt.title("차트명")  
```

축라벨과 차트 제목 크기를 변경하고 싶다면 fontsize 옵션을 사용하면 된다.


## **2. 차트추가**  

<script src="https://gist.github.com/717eunhye/a93c26f93a94ca575c433d49c8e83c8a.js"></script>

한 차트 프레임안에 두 개의 line을 그리는 경우는 plt.plot()을 추가해주면 된다.  
차트를 구분하기 위해 라벨과 색상을 추가하였고, 범례부분의 옵션을 추가로 사용해보았다. 
Matplotlib는 색상, 선크기, 선종류 등  다양하게 지원하므로 해당 문서를 참고하길 바란다. (https://wikidocs.net/92085)

![image3](https://user-images.githubusercontent.com/50131912/163089381-c1ffe644-726f-4192-b3a2-b02d38712fad.png)

```
 1) 라인색상 : plt.plot(x, y, label = 'y', color = 'red') 
 2) 라인 종류 : linestyle='dashed'  
 3) 범례종류 : plt.legend(ncol=1, fontsize = 12, frameon = True, shadow = True)  
```

**linestyle 지정하기**

Solid, Dashed, Dotted, Dash-dot과 같이 네가지 종류를 선택할 수 있다.   
![image4](https://user-images.githubusercontent.com/50131912/163089513-f8c6c6ed-5aa1-4b6c-8362-130b8ed0ef36.png)

범례 옵션별 차이점은 아래 "4.차트 프레임 2개 이상" 사용에서 비교하였다.  

**legend 위치 지정하기**  
loc 파라미터를 사용하고 숫자 튜플 또는 문자로 입력하면 된다.  
loc=(0.0, 0.0)은 데이터 영역의 왼쪽 아래, loc=(1.0, 1.0)은 데이터 영역의 오른쪽 위 이며 'best', 'upper right', 'upper left', 'lower left', 'right', 'center left', 'center right', 'lower center', 'upper center' 등으로 사용가능하다. default 값는 'best'이다. 
  
    

## **3. 차트 프레임2개 사용**  

<script src="https://gist.github.com/717eunhye/32404e17a5f31032cde28af4247a647c.js"></script>

![image5](https://user-images.githubusercontent.com/50131912/163089847-0359fc0c-b7d2-4147-a82b-233012e43ee4.png)

2번은 두 line을 한 차트에 그렸지만 개별적으로 그리고 싶을 때는 plt.subplots()를 이용하면 된다.  
plt.subplots()를 fig와 axes로 받고 첫번째 차트는 axes[0] , 두번째 차트는 axes[1]로 사용하여 각각 차트별 옵션을 다르게 설정할 수도 있고, 옵션이 동일하다면 코드의 중복을 피하기 위해 for문을 사용하는 것도 방법일 것이다. 

## **4. 차트 프레임2개 이상 사용**

<script src="https://gist.github.com/717eunhye/b344d994d5b1458f81690f89ff121893.js"></script>

![image5](https://user-images.githubusercontent.com/50131912/163090083-87887bd6-9827-4db5-bad3-567bee3a3aa1.png)

차트 프레임이 2개 이상인 경우는 axes 지정 방식이 달라진다. [행, 열]로 구분하여 차트별로 옵션 지정을 다르게 할 수 있다.  
첫번째 차트는 위의 코드와 동일하고, 두번째 차트의 경우 범례 옵션에서 frameon과 shadow를 False로 주어 테두리와 그림자를 없애주었다.  
세번째 차트는 범례를 ncol=2로 설정함으로써 범례가 가로 일직선의 형태이며  범례를 나열할 열의 갯수를 의미한다.  
마지막 네번째 차트는 두 line의 y축을 모두 표시하고 싶을 때 사용하는 방법으로 twinx() 보조축을 추가할 수 있다. 보조축을 사용하는 경우라면 아래와 같은 점을 유의해야한다. 


**매서드명 확인**  
기존에 사용하던 xlabel, ylabel, title을 그대로 사용하면 오류가 나고 앞에 set_을 붙여서 사용해야한다.  
<script src="https://gist.github.com/717eunhye/cea71fb66febd2cf29cd5acee62f476b.js"></script>

**범례**  
양쪽 두 축에 대한 범례를 하나의 상자에 표시하기 위해선 두 line을 먼저 합친 후에 legend() 메서드를 사용해야합니다. 

## **5. 수평/수직선 표시하기**

수직, 수평선을 그리는 방법은 각 2가지씩 있다.  
```
수평선 : plt.axhline(y, xmin, xmax), plt.hlines(y, xmin, xmax)  
수직선 : plt.axvline(x, ymin, ymax), plt.vlines(x, ymin, ymax)  
```
각 수평선, 수직선 함수의 차이를 비교해보도록 하자. 

<script src="https://gist.github.com/717eunhye/e1c85dca518a848d8a0a892e942930b2.js"></script>

![image6](https://user-images.githubusercontent.com/50131912/163090671-d01a399e-df86-4b7e-971e-f42e44ae9e42.png)

"3. 차트 프레임 2개 사용"한 차트에 수평/수직선을 추가해보았다. 수평선은 고정이 되는 y값을 수직선은 x 축 값을 인자로 받고 어느 정도의 범위의 선을 구할 것인지 인자로 적어주면 된다. 
	
**수평선**  
plt.axhline(고정 y축 값, 시작범위,종료범위) 인자를 받고있는데  이때 시작범위와 종료범위는 0에서 1사이의 값으로 적어주어야 하며 0은 왼쪽 끝, 1은 오른쪽 끝을 의미한다.  
plt.hlines(고정 y축 값, 시작범위,종료범위) 의 시작범위는 실제 x값의 최소, 종료범위는 실제 x값의 최대 값을 적으면된다.

**수직선**  
plt.axvline(), plt.vlines() 함수는 위의 수평선 함수와 사용법은 같다.  
  
  
  
  
  

**Reference**

[matplotlib.org](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html),
[wikidocs](https://wikidocs.net/92085), [리뷰나라](http://daplus.net/python-twinx-%EA%B0%80%EC%9E%88%EB%8A%94-%EB%B3%B4%EC%A1%B0-%EC%B6%95-%EB%B2%94%EB%A1%80%EC%97%90-%EC%B6%94%EA%B0%80%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95/)
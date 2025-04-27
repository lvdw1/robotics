#show link=underline
#align(center)[= Robotics]

== First tests
+ Training with the same reward function, but starting from downwards for swingup
  - After $10^6$ steps, he was swinging and trying to stay upwards, but not really able to stay there
  - After $3*10^6$ steps, still didn't find it
+ Modifying the reward function, based on paper
  - #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10234431")[paper]
  - Using cosine near upward position ($theta = pi$ for us)
  $
  & -7 - cos(theta) + (1-cos(theta))^3 & "if" cos(theta) < -0.76\
  & -0.2   & "else "
  $
  - 



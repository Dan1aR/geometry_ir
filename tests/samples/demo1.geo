scene "Isosceles trapezoid with circumcircle"
layout canonical=generic_auto scale=1
points A, B, C, D
trapezoid A-B-C-D [bases=A-D isosceles=true]
circle through (A, B, C, D)
target angle B-A-D
rules [no_solving=true]

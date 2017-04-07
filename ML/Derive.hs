{-
미분을 하는 함수
미분은 이산적인 계산이 아니기 때문에 아주 작은 변화량의 평균변화율을 구하여 근삿값을 구한다
-}

module ML.Derive where

derive f x = (f (x + delta) - f(x - delta)) / 2.0e-6
    where delta = 1.0e-6
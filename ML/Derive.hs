-- 테일러 급수로 미분 근사치를 계산한다
module ML.Derive where

derive f x = (f (x + delta) - f(x - delta)) / 2.0e-6
    where delta = 1.0e-6

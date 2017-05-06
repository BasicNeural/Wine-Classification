module ML.NN where

import Data.Matrix
import ML.Derive
import Data.Foldable (foldl')

-- 네트워크의 레이어를 정의
data Layer = Layer (Float -> Float) (Matrix Float) (Matrix Float)

instance Show Layer where
    show (Layer _ synapse bias) = "synapse : \n" ++ show synapse ++ "bias : \n" ++ show bias

instance Num Layer where
    (+) (Layer active lmatrix lBias) (Layer _ rmatrix rBias) = (Layer active (lmatrix + rmatrix) (lBias + rBias))
    (-) (Layer active lmatrix lBias) (Layer _ rmatrix rBias) = (Layer active (lmatrix - rmatrix) (lBias - rBias))
    (*) (Layer active lmatrix lBias) (Layer _ rmatrix rBias) = (Layer active (lmatrix * rmatrix) (lBias * rBias))
    negate      = error "ERROR!"
    abs         = error "ERROR!"
    signum      = error "ERROR!"
    fromInteger = error "ERROR!"

type Network = [Layer]

-- 뉴런 네트워크 map 함수
layerMap f (Layer active synapse bias) = (Layer active (fmap f synapse) (fmap f bias))

-- 뉴런 네트워크 순전파 함수
execute input (Layer active synapse bias) = fmap active $ synapse * input + bias

executeLayer []      input = input
executeLayer network input = foldl' execute input network

-- 활성화 함수
sigmoid x = 1 / (1 + exp (-x))

-- 로지스틱 함수
logistic d y = - d * log y - (1 - d) * log (1 - y)

-- 역전파 미분 함수
backpropagation network (x, y) = reverse . backpro (reverse network) inputs $ executeLayer network x - y
    where inputs = foldl' (\x layer -> execute (head x) layer : x ) [x] $ init network
          backpro  ((Layer active synapse bias):layers) (input:inputs) delta = 
                       (Layer active (delta * transpose input) delta) : backpro' layers inputs (transpose synapse * delta)
          backpro' []                                   _              _     = []
          backpro' ((Layer active synapse bias):layers) (input:inputs) delta = 
                    (Layer active (newDelta * transpose input) newDelta) : backpro' layers inputs (transpose synapse * newDelta)
              where newDelta = fromLists . map (:[]) . zipWith (*) (toList delta)
                               $ toList (fmap (derive active) $ synapse * input + bias)

-- 확률적 경사 하강 함수
sdg eps samples network = foldl' (\net sample -> zipWith (-) net 
        $ map (layerMap (*eps)) 
        $ backpropagation net sample) network samples

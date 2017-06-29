module ML.NN where

import Data.Matrix
import ML.Derive
import qualified Data.Vector as V
import System.Rsndom

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

-- 뉴런 네트워크 map 함수
layerMap f (Layer active synapse bias) = (Layer active (fmap f synapse) (fmap f bias))

-- 뉴런 네트워크 순전파 함수
execute input (Layer active synapse bias) = fmap active $ synapse * input + bias

executeLayer network input = V.foldl' execute input network

-- 활성화 함수
sigmoid x = 1 / (1 + exp (-x))

-- 로지스틱 함수
logistic d y = - d * log y - (1 - d) * log (1 - y)

-- 역전파 미분 함수
backpropagation network (x, y) = V.reverse . backpro (V.reverse network) inputs $ executeLayer network x - y
  where inputs = V.foldl' (\x layer -> execute (V.head x) layer `V.cons` x ) (V.singleton x) $ V.init network
        backpro layer' input' delta = 
          (Layer active (delta * transpose input) delta) 
            `V.cons` backpro' layers inputs (transpose synapse * delta)
          where input  = V.head input'
                inputs = V.tail input'
                (Layer active synapse bias) = V.head layer'
                layers = V.tail layer'
                backpro' layer' input' delta =
                  (Layer active (newDelta * transpose input) newDelta)
                    `V.cons` if inputs == V.empty 
                      then V.empty
                      else backpro' layers inputs (transpose synapse * newDelta)
                  where input  = V.head input'
                        inputs = V.tail input'
                        (Layer active synapse bias) = V.head layer'
                        layers = V.tail layer'
                        newDelta = colVector . V.zipWith (*) (getMatrixAsVector delta)
                          $ getMatrixAsVector (fmap (derive active) $ synapse * input + bias)

-- 확률적 경사 하강 함수
sdg eps dataset samples network = V.foldl' (\net sample -> V.zipWith (-) net 
        $ fmap (layerMap (*eps)) 
        $ backpropagation net (dataset V.! sample)) network samples

sdg2 seed eps dataset samples network = 

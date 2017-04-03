module ML.NN where

import Data.Matrix
import ML.Derive
import Data.List (foldl')
import Control.Monad.State


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

layerMap f (Layer active synapse bias) = (Layer active (fmap f synapse) (fmap f bias))

toScalar = head . toList

execute input (Layer active synapse bias) = fmap active $ synapse * input + bias

executeLayer []      input = input
executeLayer network input = foldl' execute input network

sigmoid x = 1 / (1 + exp (-x))

reLU x = max 0 x

logistic d y = - d * log y - (1 - d) * log y

backpropagation network (x, y) = reverse . backpro (reverse network) inputs $ executeLayer network x - y
    where inputs = foldl' (\x layer -> execute (head x) layer : x ) [x] $ init network
          backpro  ((Layer active synapse bias):layers) (input:inputs) delta = 
                       (Layer active (delta * transpose input) delta) : backpro' layers inputs (transpose synapse * delta)
          backpro' []                                   _              _     = []
          backpro' ((Layer active synapse bias):layers) (input:inputs) delta = 
                       (Layer active (newDelta * transpose input) newDelta) : backpro' layers inputs (transpose synapse * newDelta)
              where newDelta = fromLists . map (:[]) . zipWith (*) (toList delta)
                               $ toList (fmap (derive active) $ synapse * input + bias)
{-
sdg eps step network dataset = loop step network
        where loop n network | n /= 0    = do rand <- randomRIO (0, length dataset - 1)
                                              loop (n - 1) (zipWith (-) network (map (layerMap (*eps)) (backpropagation network (dataset !! fromIntegral rand))) )
                             | otherwise = return network
-}
sdg eps samples network dataset = foldl' (\net sample -> zipWith (-) net 
        $ map (layerMap (*eps)) 
        $ backpropagation net $ dataset !! sample) network samples
{-
sdg eps samples network dataset = execState 
    (mapM_ (\sample -> 
        modify (\net ->
            zipWith (-) net . map (layerMap (*eps))
            . backpropagation net $ dataset !! sample)) samples) network
-}
module Main where

import ML.NN
import System.Random
import Data.Matrix                          as M
import qualified Data.Sequence              as S
import qualified Data.Foldable              as F
import qualified Data.ByteString.Lazy.Char8 as BS

makeRand x = replicate x $ randomRIO (0.0, 1.0)

oneShot x = fromList len 1 $ replicate r 0 ++ [1] ++ iterate (\x -> x) 0
    where r = snd . maximum $ zip (toList x) [0 .. length x - 1]
          len = length x

main = do
    rawdata <- fmap BS.lines $ BS.readFile ".\\resource\\wine_train.csv"

    let set = map (map (\x -> read x :: Float) . map BS.unpack . BS.split ',') rawdata

    let x = map (transpose . fromLists . (:[]) . take 13) set
    let y = map (transpose . fromLists . (:[]) . drop 13) set

    seed1 <- sequence $ makeRand 52
    seed2 <- sequence $ makeRand 12
    let s1 = fromList 4 13 seed1
    let b1 = fromList 4 1 $ replicate 4 0
    let s2 = fromList 3 4 $ seed2
    let b2 = fromList 3 1 $ replicate 3 0
    let h1 = (Layer sigmoid s1 b1)
    let h2 = (Layer sigmoid s2 b2)
    let dataset = zip x y

    samplesindex <- sequence . replicate 200000 $ randomRIO (0, length dataset - 1)

    let samples = map (\x -> dataset !! x) samplesindex

    let result = sdg 0.1 samples [h1,h2]

    testdata <- fmap BS.lines $ BS.readFile ".\\resource\\wine_test.csv"

    let testSet = map (map (\x -> read x :: Float) . map BS.unpack . BS.split ',') testdata

    let testX = map (transpose . fromLists . (:[]) . take 13) testSet
    let testY = map (transpose . fromLists . (:[]) . drop 13) testSet

    putStr "테스트 오차율 : "

    putStr . show . (*100) . (/ fromIntegral (length testY)) . sum
           . map (\x -> if x then 0 else 1)
           . zipWith (==) testY $ map (oneShot . executeLayer result) testX

    putStrLn "%"
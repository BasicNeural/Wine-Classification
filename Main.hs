module Main where

import ML.NN
import System.Random
import Data.Matrix                          as M
import qualified Data.Sequence              as S
import qualified Data.Foldable              as F
import qualified Data.ByteString.Lazy.Char8 as BS

-- 랜덤 값 생성
makeRand x = replicate x $ randomRIO (0.0, 1.0)

-- 행렬을 받아 가장 큰 값만 1로 만들고 나머지를 0으로 만드는 함수
oneShot x = fromList len 1 $ replicate r 0 ++ [1] ++ iterate (\x -> x) 0
    where r = snd . maximum $ zip (toList x) [0 .. length x - 1]
          len = length x

main = do
    -- 학습용 데이터를 불러옴
    rawdata <- fmap BS.lines $ BS.readFile ".\\resource\\wine_train.csv"

    let set = map (map (\x -> read x :: Float) . map BS.unpack . BS.split ',') rawdata
    
    -- 입력 리스트
    let x = map (transpose . fromLists . (:[]) . take 13) set
    
    -- 목표출력 리스트
    let y = map (transpose . fromLists . (:[]) . drop 13) set

    -- 뉴런 네트워크 생성
    seed1 <- sequence $ makeRand 52
    seed2 <- sequence $ makeRand 12
    let s1 = fromList 4 13 seed1
    let b1 = fromList 4 1 $ replicate 4 0
    let s2 = fromList 3 4 $ seed2
    let b2 = fromList 3 1 $ replicate 3 0
    let h1 = (Layer sigmoid s1 b1)
    let h2 = (Layer sigmoid s2 b2)
    let dataset = zip x y

    -- 랜덤하게 샘플 선정
    samplesindex <- sequence . replicate 80000 $ randomRIO (0, length dataset - 1)

    -- 샘플 데이터 리스트 생성
    let samples = map (\x -> dataset !! x) samplesindex

    -- 확률적 경사 하강법으로 학습
    let result = sdg 0.1 samples [h1,h2]

    -- 테스트용 데이터를 불러옴
    testdata <- fmap BS.lines $ BS.readFile ".\\resource\\wine_test.csv"

    let testSet = map (map (\x -> read x :: Float) . map BS.unpack . BS.split ',') testdata

    -- 입력 리스트
    let testX = map (transpose . fromLists . (:[]) . take 13) testSet
    -- 목표 출력 리스트
    let testY = map (transpose . fromLists . (:[]) . drop 13) testSet

    putStr "테스트 오차율 : "

    putStr . show . (*100) . (/ fromIntegral (length testY)) . sum
           . map (\x -> if x then 0 else 1)
           . zipWith (==) testY $ map (oneShot . executeLayer result) testX

    putStrLn "%"
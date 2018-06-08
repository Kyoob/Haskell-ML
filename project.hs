{-
    CS457 Project: A perceptron built to perform binary classification on two
    different datasets. One is a hill/valley dataset, where 100 points on a
    graph either create what looks like a dip (valley, 0) or bump (hill, 1).
    The other is a breast cancer dataset, where 9 different attributes signify
    whether a tumor is benign (2) or malignant (4). The user can select which
    one they wish to classify.
    For the hill/valley dataset, you can change the files used in getData from
    HVTrain and HVTest to HVTrainN and HVTestN to use the examples that contain
    noise, wherein the hills and valleys created are not nearly as smooth
    looking. As expected, the accuracy on these examples is slightly worse.
    Prof. Tolmach
    Tim Coutinho
-}

import Control.Monad (when)
import Data.List
import System.IO
import System.Random hiding (split)


main :: IO ()
main = do putStr "Enter HV to use the hill/valley dataset, and anything else "
          putStr "to use the breast cancer dataset: "; hFlush stdout
          d <- getLine
          (xTrain, yTrain, xTest, yTest) <- getData d
          (xTrain, xTest) <- return $ (normalize xTrain, normalize xTest)
          putStr "Enter learning rate (0.1 is a good default): "; hFlush stdout
          lr <- getLine
          putStr "Enter number of epochs to train: "; hFlush stdout
          epochs <- getLine
          g <- getStdGen
          weights <- getWeights (length (head xTest)) g
          weights <- train (read epochs :: Int) (read epochs :: Int)
                           (read lr :: Float) xTrain yTrain xTest yTest weights
          putStrLn "The model is now trained, and can make predictions."
          putStrLn "For the hill/valley dataset, input lists of 100 floats."
          putStrLn "For the breast cancer dataset, input lists of 9 integers."
          putStrLn "An output of 0 is valley/benign, and 1 is hill/malignant."
          putStrLn "Type q to quit."
          doPredictions weights d
          putStrLn "Farewell."

-- Trains the weights using the training data (xs ys) and tests the accuracy on
-- both the testing data (xs' ys') and training data
train :: (Ord a, Fractional a) => Int -> Int -> a -> [[a]] -> [Int]
                               -> [[a]] -> [Int] -> [a] -> IO ([a])
train i j lr xs ys xs' ys' w
 | i == 0    = do putStrLn $ "Epoch "  ++ show j ++ ":"
                  putStrLn $ "Train: " ++ show (accuracy xs ys w)
                  putStrLn $ "Test: "  ++ show (accuracy xs' ys' w)
                  return w
 | otherwise = do putStrLn $ "Epoch "  ++ show (j-i) ++ ":"
                  putStrLn $ "Train: " ++ show (accuracy xs ys w)
                  putStrLn $ "Test: "  ++ show (accuracy xs' ys' w)
                  predictions <- return $ predict xs w
                  errorVals   <- return $ errors lr xs ys predictions
                  newWeights  <- return $ update w errorVals
                  train (i-1) j lr xs ys xs' ys' newWeights

-- Reads the data from the corresponding file, depending on user input
getData :: String -> IO ([[Float]], [Int], [[Float]], [Int])
getData s
 | s == "HV" = do training <- readFile "data/HVTrain.csv"
                  testing  <- readFile "data/HVTest.csv"
                  return (images training, labels training,
                          images testing, labels testing)
 | otherwise = do training <- readFile "data/BreastCancerTrain.csv"
                  testing  <- readFile "data/BreastCancerTest.csv"
                  return (images' training, labels' training,
                          images' testing, labels' testing)
  where labels  = map ((\s -> read s :: Int) . takeWhile (/= '\r') . last) . parseCSV
        images  = map (map (\s -> read s :: Float) . init) . parseCSV
        labels' = map ((\s -> (read s :: Int) // 2 - 1) . last) . parseCSV
        images' = map (map (\s -> read s :: Float) . init . tail) . parseCSV

-- Converts a .csv file into a list of lists, delimited at newlines and commas
parseCSV :: String -> [[String]]
parseCSV = map (split ',') . split '\n'

-- Splits a list at a given delimiter, not sure how vanilla Haskell doesn't
-- have a similiar function in the Prelude
split :: (Eq t) => t -> [t] -> [[t]]
split c [] = []
split c s  = takeWhile (/= c) s : split c (tail' $ dropWhile (/= c) s)
 where tail' []     = []
       tail' (_:xs) = xs

-- Generates the given number of random weights (plus one for the bias) ranging
-- from -0.05 to 0.05
getWeights :: (Fractional a, Random a) => Int -> StdGen -> IO [a]
getWeights n = return . take (n+1) . randomRs (-0.05, 0.05)

-- Normalizes the values of a dataset. Normalization method changes depending
-- on the dataset used, due to the way the values in each one work
normalize :: (Ord a, Fractional a) => [[a]] -> [[a]]
normalize l
 | length (head l) == 100 = map normalize' l
 | otherwise = (transpose . map normalize' . transpose) l
 where normalize' l'   = map (\x -> (x - lo)/(hi - lo)) l'
        where (hi, lo) = (maximum l', minimum l')

-- Vector/matrix multiplication
dotM :: Num a => [[a]] -> [a] -> [a]
dotM xs ys = zipWith dot xs (repeat ys)
 where dot xs' ys' = sum $ zipWith (*) xs' ys'

-- Calculates the accuracy on a given set of inputs, outputs, and weights
accuracy :: (Ord a, Num a, Fractional b) => [[a]] -> [Int] -> [a] -> b
accuracy xs ys w = trunc 3 $ count correct / n * 100
 where count     = fromIntegral . length . filter (== True)
       correct   = zipWith (==) ys (predict xs w)
       n         = fromIntegral $ length ys
       trunc i x = (fromInteger $ round $ x * (10^i)) / 10.0^^i

-- Predicts the output on a given set of inputs and weights. A one is prepended
-- to each input for the bias unit
predict :: (Ord a, Num a) => [[a]] -> [a] -> [Int]
predict xs w = map (fromEnum . (> 0)) $ (map (1:) xs) `dotM` w

-- Calculates the average error for each input feature on a given set of
-- inputs, outputs, and predictions
errors :: (Integral a, Fractional b) => b -> [[b]] -> [a] -> [a] -> [b]
errors lr xs ys preds = (map avg . transpose) $ zipWith (\d x -> map (*d) x) difs xs
 where difs  = zipWith (\y p -> lr * fromIntegral (y-p)) ys preds
       avg l = (sum l) / (fromIntegral (length l))

-- Updates a given set of weights by adding the corresponding (negative) error
-- on each weight
update :: (Num a) => [a] -> [a] -> [a]
update w e = zipWith (+) w e

-- Allows user to input a list of values and have the output predicted. Awkward
-- steps have to be taken to re-normalize the data using the added value
doPredictions :: [Float] -> String -> IO ()
doPredictions w d = do p <- getLine
                       (xs, _, _, _) <- getData d
                       when (p /= "q") $ do
                       print $ predict [head (normalize ((read p :: [Float]) : xs))] w
                       doPredictions w d

-- I just like this better than div
(//) :: Integral a => a -> a -> a
(//) x y = x `div` y

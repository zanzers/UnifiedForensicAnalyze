using System;
using System.Collections.Generic;
using System.IO;
using OpenCvSharp;
using System.Diagnostics;
using System.Text.Json.Nodes;



namespace UnifiedForensicsAnalyze.Features
{
    public class ELAStage : IAnalysisStage
    {
        public string Name => "ELA";
        

        public StageResult Process(Mat? input)
        {
            Console.WriteLine($"ELA On Prosses");
            Mat IinFloat = new();
            input.ConvertTo(IinFloat, MatType.CV_32FC3, 1.0 / 255.0);

            Cv2.ImEncode(".jpg", input, out byte[] jpegBytes, new int[] { (int)ImwriteFlags.JpegQuality, 95 });
            Mat jpegImg = Cv2.ImDecode(jpegBytes, ImreadModes.Color);
            jpegImg.ConvertTo(jpegImg, MatType.CV_32FC3, 1.0 / 255.0);

            Mat ela = new();
            Cv2.Absdiff(IinFloat, jpegImg, ela);


            Mat elaGray = new();
            Cv2.CvtColor(ela, elaGray, ColorConversionCodes.BGR2GRAY);

            elaGray.GetArray(out float[] elaData);
            double[] data = Array.ConvertAll(elaData, x => (double)x);


            double mean = data.Average();
            double variance = data.Select(v => Math.Pow(v - mean, 2)).Average();
            double stddev = Math.Sqrt(variance);

            int bins = 256;
            double[] hist = new double[bins];
            foreach (var v in data)
            {
                int idx = (int)(v * (bins - 1));
                hist[idx]++;
            }
            hist = hist.Select(h => h / data.Length).ToArray();
            double entropy = -hist.Where(p => p > 0).Sum(p => p * Math.Log(p, 2));


            double skewness = data.Select(v => Math.Pow((v - mean) / stddev, 3)).Average();
            double kurtosis = data.Select(v => Math.Pow((v - mean) / stddev, 4)).Average();



            Mat elaoutput = new Mat();
            Cv2.Normalize(ela, elaoutput, 0, 255, NormTypes.MinMax);
            elaoutput.ConvertTo(elaoutput, MatType.CV_8UC3);

            // Extraction data!
            var features = new Dictionary<string, double>
            {
                { "Ela_Mean", mean },
                { "Ela_StdDev", stddev },
                { "Ela_Entropy", entropy },
                { "Ela_Skewness", skewness },
                { "Ela_Kurtosis", kurtosis }
            };

            Console.WriteLine($"ELA Done");
            return new StageResult
            {
                Features = features
            };
        }
    }

    public class SVDStage : IAnalysisStage
    {
    public string Name => "SVD";

        public StageResult Process(Mat? input)
        {
            if (input == null || input.Empty())
                throw new ArgumentException("SVDStage received invalid input image.");

            Console.WriteLine($"SVD processing...");

            int maxSide = 512;
            Mat resizedInput = input.Clone();
            if (Math.Max(input.Width, input.Height) > maxSide)
            {
                double scale = (double)maxSide / Math.Max(input.Width, input.Height);
                int newW = (int)(input.Width * scale);
                int newH = (int)(input.Height * scale);
                Cv2.Resize(input, resizedInput, new Size(newW, newH));
                Console.WriteLine($"SVD resized to: {newW}x{newH}");
            }

            Mat gray = new();
            Cv2.CvtColor(resizedInput, gray, ColorConversionCodes.BGR2GRAY);
            gray.ConvertTo(gray, MatType.CV_32F);

            Mat W = new();
            Mat U = new();
            Mat Vt = new();
            Cv2.SVDecomp(gray, W, U, Vt);

            float[] svArray = new float[W.Rows * W.Cols];
            W.GetArray(out svArray);

            double eps = 1e-12;
            double[] svDouble = svArray.Select(v => Math.Abs((double)v)).ToArray();
            double sumSv = svDouble.Sum() + eps;
            double[] svNorm = svDouble.Select(v => v / sumSv).ToArray();


            double meanSV = svDouble.Average();
            double stdSV = Math.Sqrt(svDouble.Select(v => Math.Pow(v - meanSV, 2)).Average());
            double top1 = svNorm.Length > 0 ? svNorm[0] : 0;
            double top5 = svNorm.Take(Math.Min(5, svNorm.Length)).Sum();
            double top10 = svNorm.Take(Math.Min(10, svNorm.Length)).Sum();
            double entropy = -svNorm.Sum(p => p > 0 ? p * Math.Log(p + eps) : 0);

            var features = new Dictionary<string, double>
            {
                {"SVD_Top1", top1},
                {"SVD_Top5", top5},
                {"SVD_Top10", top10},
                {"SVD_Mean", meanSV},
                {"SVD_stdSV", stdSV},
                {"SVD_Entropy", entropy}
            };

            Console.WriteLine($"SVD done.");
            return new StageResult
            {
                 // skip visualization
                Features = features
            };
        }
    }

    public class IWTStage : IAnalysisStage
    {
        public string Name => "IWT";

        public StageResult Process(Mat? input)
        {
            Console.WriteLine($"IWT On Prosses");

            Mat gray = new Mat();
            Cv2.CvtColor(input, gray, ColorConversionCodes.BGR2GRAY);
            gray.ConvertTo(gray, MatType.CV_32F);

            int rows = gray.Rows;
            int cols = gray.Cols;


            int rowsEven = rows - (rows % 2);
            int colsEven = cols - (cols % 2);
            Mat resized = gray.SubMat(0, rowsEven, 0, colsEven);


            Mat LL = new Mat(rowsEven / 2, colsEven / 2, MatType.CV_32F);
            Mat LH = new Mat(rowsEven / 2, colsEven / 2, MatType.CV_32F);
            Mat HL = new Mat(rowsEven / 2, colsEven / 2, MatType.CV_32F);
            Mat HH = new Mat(rowsEven / 2, colsEven / 2, MatType.CV_32F);


            for (int y = 0; y < rowsEven; y += 2)
            {
                for (int x = 0; x < colsEven; x += 2)
                {
                    float a = resized.At<float>(y, x);
                    float b = resized.At<float>(y, x + 1);
                    float c = resized.At<float>(y + 1, x);
                    float d = resized.At<float>(y + 1, x + 1);


                    LL.Set<float>(y / 2, x / 2, (a + b + c + d) / 4f);
                    LH.Set<float>(y / 2, x / 2, (a + b - c - d) / 4f);
                    HL.Set<float>(y / 2, x / 2, (a - b + c - d) / 4f);
                    HH.Set<float>(y / 2, x / 2, (a - b - c + d) / 4f);
                }
            }


            double[] ll = ComputeStats(LL);
            double[] lh = ComputeStats(LH);
            double[] hl = ComputeStats(HL);
            double[] hh = ComputeStats(HH);

            Mat top = new Mat();
            Mat bottom = new Mat();
            Mat output = new Mat();
            
            Cv2.Normalize(output, output, 0, 255, NormTypes.MinMax);
            output.ConvertTo(output, MatType.CV_8UC1);


            double[] featureVector = new double[]
            {
            ll[0], ll[1], ll[2],
            lh[0], lh[1], lh[2],
            hl[0], hl[1], hl[2],
            hh[0], hh[1], hh[2]
            };

            var features = new Dictionary<string, double>
            {
                { "IWT_LL_Mean", ll[0] },
                { "IWT_LL_Var", ll[1] },
                { "IWT_LL_Entropy", ll[2] },

                { "IWT_LH_Mean", lh[0] },
                { "IWT_LH_Var", lh[1] },
                { "IWT_LH_Entropy", lh[2] },

                { "IWT_HL_Mean", hl[0] },
                { "IWT_HL_Var", hl[1] },
                { "IWT_HL_Entropy", hl[2] },

                { "IWT_HH_Mean", hh[0] },
                { "IWT_HH_Var", hh[1] },
                { "IWT_HH_Entropy", hh[2] }
            };

            Console.WriteLine($"IWT Done");
            return new StageResult
            {
                Features = features
            };
        }
        private double[] ComputeStats(Mat m)
        {
            Scalar mean, stddev;
            Cv2.MeanStdDev(m, out mean, out stddev);

            double meanVal = mean.Val0;
            double varVal = stddev.Val0 * stddev.Val0;


            int bins = 256;
            Rangef range = new Rangef(0, 256);
            Mat hist = new Mat();
            Cv2.CalcHist(new Mat[] { m }, new int[] { 0 }, null, hist, 1, new int[] { bins }, new Rangef[] { range });
            hist /= m.Total();

            double entropy = 0.0;
            for (int i = 0; i < bins; i++)
            {
                double p = hist.At<float>(i);
                if (p > 0)
                    entropy -= p * Math.Log(p, 2);
            }

            return new double[] { meanVal, varVal, entropy };
        }
    }


    public class PRNUStage : IAnalysisStage
    {
        public string Name => "PRNU";

        public StageResult Process(Mat? input)
        {
            if (input == null || input.Empty())
                throw new ArgumentException("PRNUStage received invalid input image.");

            Console.WriteLine("PRNU On Process");

            // Convert to float RGB
            Mat Igray = new();
            Cv2.CvtColor(input, Igray, ColorConversionCodes.BGR2RGB);
            Igray.ConvertTo(Igray, MatType.CV_32FC3, 1.0 / 255.0);

            // Denoise
            Mat Iden8 = new();
            Cv2.FastNlMeansDenoisingColored(input, Iden8, 10, 10, 7, 21);
            Mat Iden = new();
            Iden8.ConvertTo(Iden, MatType.CV_32FC3, 1.0 / 255.0);

            // PRNU extraction
            Mat W = new();
            Cv2.Subtract(Igray, Iden, W);

            // Normalize safely
            Mat denom = new();
            Cv2.Multiply(Igray, Igray, denom);
            Mat denomSafe = new();
            Cv2.Add(denom, new Scalar(1e-8f, 1e-8f, 1e-8f), denomSafe);

            Mat num = new();
            Cv2.Multiply(W, Igray, num);

            Mat K = new();
            Cv2.Divide(num, denomSafe, K);
            Cv2.PatchNaNs(K, 0);

            // Zero-mean
            Scalar mean = Cv2.Mean(K);
            Mat Kzm = new();
            Cv2.Subtract(K, new Scalar(mean.Val0, mean.Val1, mean.Val2), Kzm);

            // Normalize
            Mat Ksq = new();
            Cv2.Pow(Kzm, 2, Ksq);
            double norm = Math.Sqrt(Cv2.Sum(Ksq).Val0) + 1e-8;
            Mat Knorm = new();
            Cv2.Divide(Kzm, new Scalar((float)norm, (float)norm, (float)norm), Knorm);
            Cv2.PatchNaNs(Knorm, 0);

            // --- Block-based features ---
            int blockSize = 32;
            List<double> blockVars = new();
            List<double> blockMeans = new();
            List<double> blockEnergies = new();

            for (int y = 0; y < Knorm.Rows; y += blockSize)
            {
                for (int x = 0; x < Knorm.Cols; x += blockSize)
                {
                    Rect roi = new(x, y, Math.Min(blockSize, Knorm.Cols - x), Math.Min(blockSize, Knorm.Rows - y));
                    Mat block = new(Knorm, roi);
                    Cv2.MeanStdDev(block, out Scalar blkMean, out Scalar blkStd);
                    double energy = Cv2.Norm(block, NormTypes.L2);

                    blockMeans.Add((blkMean.Val0 + blkMean.Val1 + blkMean.Val2) / 3.0);
                    blockVars.Add((blkStd.Val0 + blkStd.Val1 + blkStd.Val2) / 3.0);
                    blockEnergies.Add(energy);
                }
            }

            // --- Wavelet-domain features ---
            // --- Wavelet-domain proxy ---
            Mat gray = new();
            Cv2.CvtColor(input, gray, ColorConversionCodes.BGR2GRAY);
            gray.ConvertTo(gray, MatType.CV_32F, 1.0 / 255.0);

            Mat LL = new();
            Cv2.PyrDown(gray, LL);                  // low-frequency component

            Mat LL_up = new();
            Cv2.PyrUp(LL, LL_up, new Size(gray.Cols, gray.Rows)); // upsample LL back

            Mat HF = new();
            Cv2.Subtract(gray, LL_up, HF);          // high-frequency component

            double waveletEnergyLL = Cv2.Norm(LL, NormTypes.L2);
            double waveletEnergyHF = Cv2.Norm(HF, NormTypes.L2);
            double waveletRatio = waveletEnergyHF / (waveletEnergyLL + 1e-8);

            // --- Flatten K for global stats ---
            Mat flat = Knorm.Reshape(1, 1);
            flat.GetArray(out float[] arr);
            for (int i = 0; i < arr.Length; i++)
                if (float.IsNaN(arr[i]) || float.IsInfinity(arr[i]))
                    arr[i] = 0;

            Cv2.MeanStdDev(flat, out Scalar fMean, out Scalar fStd);
            double energyTotal = Cv2.Norm(flat, NormTypes.L2);

            // --- Compose features dictionary ---
            var features = new Dictionary<string, double>
            {
                { "PRNU_GMean", fMean.Val0 },
                { "PRNU_GStdDev", fStd.Val0 },
                { "PRNU_GEnergy", energyTotal },
                { "PRNU_BMean", blockMeans.Average() },
                { "PRNU_BStd", blockVars.Average() },
                { "PRNU_BEnergy", blockEnergies.Average() },
                { "PRNU_BMax", blockMeans.Max() },
                { "PRNU_B_Std_Max", blockVars.Max() },
                { "Wavelet_Energy_LL", waveletEnergyLL },
                { "Wavelet_Energy_HF", waveletEnergyHF },
                { "Wavelet_HF_to_LL_Ratio", waveletRatio }
            };

            Console.WriteLine("PRNU Done");
            return new StageResult { Features = features };
        }
    }



    public class CnnStage : IAnalysisStage
    {
        public string Name => "CNN_Model";

        public StageResult Process(Mat? input)
        {
            try
            {

                string uploadDir = Path.Combine("uploads");
                string[] files = Directory.GetFiles(uploadDir);
                string inputFilePath = Path.GetFullPath(files[0]);

                //  ! Uncommit in case of extracting Data.....
                // string inputFilePath = Bypass.GetPath();
                Console.WriteLine($"CNN Recied Input Path: {inputFilePath}");

                string output = PythonRunner.Run("cnn_model.py",inputType: "s", inputFilePath);

                string jsonLine = null;
                foreach (var line in output.Split('\n'))
                {
                    string trimmed = line.Trim();
                    if (trimmed.StartsWith("{") && trimmed.EndsWith("}"))
                    {
                        jsonLine = trimmed;
                        break;
                    }
                }

                if (jsonLine == null)
                {
                    Console.WriteLine("[CNN model] Invalid JSON found in Python output.");
                    return new StageResult { Success = false, Output = "Invalid JSON from Python" };
                }


                var parsed = JsonNode.Parse(jsonLine);
                var featuresCnn = new Dictionary<string, double>();

                int label = parsed["CNN_label"]?.GetValue<int>() ?? -1;

                if(label == -1) label = 1;
                double confidence = parsed["CNN_confidence"]?.GetValue<double>() ?? 0.0;

                var probArray = parsed["CNN_probabilities"]?.AsArray();
                double prob0 = probArray?.Count > 0 ? probArray[0]!.GetValue<double>() : 0;
                double prob1 = probArray?.Count > 1 ? probArray[1]!.GetValue<double>() : 0;
                double prob2 = probArray?.Count > 2 ? probArray[2]!.GetValue<double>() : 0;

                featuresCnn["CNN_Label"] = label;
                featuresCnn["CNN_Confidence"] = confidence;
                featuresCnn["CNN_Prob_0"] = prob0;
                featuresCnn["CNN_Prob_1"] = prob1;
                featuresCnn["CNN_Prob_2"] = prob2;



                Console.WriteLine($"CNN Done");
                return new StageResult
                {
                    Features = featuresCnn,
                    Output = "[CNN Stage Completed]",
                    Success = true
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[CNN model] Error: {ex.Message}");
                return new StageResult
                {
                    Output = ex.Message,
                    Success = false
                };
            }
        }


    }


    

}





    // extraction clip() path:

    // lstm.py ->return path:?
    // predict.py path-> result ... || YOU!!!!!!!!.
    // save output....16pcs?




    // images run() -> 16pcs 
    // 
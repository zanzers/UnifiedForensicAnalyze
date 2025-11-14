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


            return new StageResult
            {
                OutputImage = elaoutput,
                Features = features

            };
        }
    }

    public class SVDStage : IAnalysisStage
    {
        public string Name => "SVD";


        public StageResult Process(Mat? input)
        {
            Mat gray = new();
            Cv2.CvtColor(input, gray, ColorConversionCodes.BGR2GRAY);
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

            // Compute Features
            double totalSV = svDouble.Length;
            double meanSV = svDouble.Average();
            double stdSV = Math.Sqrt(svDouble.Select(v => Math.Pow(v - meanSV, 2)).Average());
            double top1 = svNorm.Length > 0 ? svNorm[0] : 0;
            double top5 = svNorm.Take(Math.Min(5, svNorm.Length)).Sum();
            double top10 = svNorm.Take(Math.Min(10, svNorm.Length)).Sum();
            double entropy = -svNorm.Sum(p => p > 0 ? p * Math.Log(p + eps) : 0);



            Mat woutput = new Mat(W.Rows, W.Cols, MatType.CV_8UC1);
            Cv2.Normalize(W, woutput, 0, 255, NormTypes.MinMax);
            woutput.ConvertTo(woutput, MatType.CV_8UC1);


            var features = new Dictionary<string, double>
            {
                {"SVD_Top1", top1},
                {"SVD_Top5", top5},
                {"SVD_Top10", top10},
                {"SVD_Mean", meanSV},
                {"SVD_stdSV", stdSV},
                {"SVD_Entropy", entropy}

            };

            return new StageResult
            {
                OutputImage = woutput,
                Features = features

            };
        }
    }

    public class IWTStage : IAnalysisStage
    {
        public string Name => "IWT";

        public StageResult Process(Mat? input)
        {

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
            Cv2.HConcat(new Mat[] { LL, LH }, top);
            Cv2.HConcat(new Mat[] { HL, HH }, bottom);
            Cv2.VConcat(new Mat[] { top, bottom }, output);

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

            return new StageResult
            {
                OutputImage = output,
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
        private Mat? numeratorSum;
        private Mat? denominatorSum;
        private int imageCount = 0;

        public StageResult Process(Mat? input)
        {
            Mat Iout = new();
            Cv2.CvtColor(input, Iout, ColorConversionCodes.BGR2RGB);
            Iout.ConvertTo(Iout, MatType.CV_32FC3, 1.0 / 255.0);

            Mat Iin = new();
            Cv2.FastNlMeansDenoisingColored(input, Iin, 10, 10, 7, 21);
            Iin.ConvertTo(Iin, MatType.CV_32FC3, 1.0 / 255.0);

            Mat W = Iout - Iin;

            Mat num = new();
            Cv2.Multiply(W, Iin, num);


            if (numeratorSum == null)
            {
                numeratorSum = Mat.Zeros(input.Size(), MatType.CV_32FC3);
                denominatorSum = Mat.Zeros(input.Size(), MatType.CV_32FC3);
            }

            Cv2.Add(numeratorSum, num, numeratorSum);
            Cv2.Add(denominatorSum, num, denominatorSum);
            imageCount++;

            Mat eps = Mat.Ones(denominatorSum.Size(), denominatorSum.Type()) * 1e-8f;
            Mat denomSafe = denominatorSum + eps;

            Mat F = new();
            Cv2.Divide(numeratorSum, denomSafe, F);
            Cv2.PatchNaNs(F, 0);

            Scalar mean = Cv2.Mean(F);
            Mat FzeroMean = new();
            Cv2.Subtract(F, new Scalar(mean.Val0, mean.Val1, mean.Val2), FzeroMean);

            Mat sq = new();
            Cv2.Pow(FzeroMean, 2.0, sq);
            double norm = Math.Sqrt(Cv2.Sum(sq).Val0) + 1e-8;
            if (double.IsNaN(norm) || norm < 1e-8) norm = 1e-8;




            Mat Fnormalized = FzeroMean / norm;
            Cv2.PatchNaNs(Fnormalized, 0);

            Mat Foutput = new();
            Cv2.Normalize(Fnormalized, Foutput, 0, 255, NormTypes.MinMax);
            Foutput.ConvertTo(Foutput, MatType.CV_8UC1);


            Mat features = Fnormalized.Reshape(1, 1);
            features.GetArray(out float[] data);

            for (int i = 0; i < data.Length; i++)
            {
                if (float.IsNaN(data[i]) || float.IsInfinity(data[i])) data[i] = 0f;
            }

            Scalar meanOut, stddev;
            Cv2.MeanStdDev(features, out meanOut, out stddev);
            double energy = Cv2.Norm(features, NormTypes.L2);

            Cv2.MeanStdDev(FzeroMean, out Scalar preMean, out Scalar preStd);


            var featuresPrnu = new Dictionary<string, double>
            {
                { "PRNU_Mean", meanOut.Val0 },
                { "PRNU_Normal_Mean", preMean.Val0 },
                { "PRNU_Normal_Std", preStd.Val0 },
                { "PRNU_StdDev", stddev.Val0 },
                { "PRNU_Energy", energy }
            };

            return new StageResult
            {
                OutputImage = Foutput,
                Features = featuresPrnu
            };

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

                Console.WriteLine($"CNN call by C: {inputFilePath}");

                string output = PythonRunner.Run("cnn_model.py", inputFilePath);
                Console.WriteLine($"[CNN model] Python returned: \n{output}");


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



    public class Extraction : IAnalysisStage
{
    public string Name => "Video_Extraction";
    private readonly string _videoPath;

    public Extraction(string videoPath)
    {
        if (string.IsNullOrWhiteSpace(videoPath))
            throw new ArgumentNullException(nameof(videoPath));

        _videoPath = videoPath;
    }

   public StageResult Process(Mat? input) 
    {
        try
        {
            Console.WriteLine("[Extraction] Starting video processing...");

            string baseDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "ExtractedData");
            string clipsDir = Path.Combine(baseDir, "clips");
            string frameDir = Path.Combine(baseDir, "frames");
            Directory.CreateDirectory(clipsDir);
            Directory.CreateDirectory(frameDir);

            int framesPerClip = 16;
            using var capture = new VideoCapture(_videoPath);
            if(!capture.IsOpened()) throw new Exception($"Cannot open video file: {_videoPath}");

            double fps = capture.Fps;
            int width = (int)capture.FrameWidth;
            int height = (int)capture.FrameHeight;
            int totalFrames = (int)capture.FrameCount;

            Console.WriteLine($"[INFO]: {totalFrames} frames, {fps:F2} FPS, Resolution: {width}x{height}");

            int frameCount = 0;
            int clipCount = 0;
            var framesBuffer = new List<Mat>();
            var frame = new Mat();

            while (true)
            {
                capture.Read(frame);
                if(frame.Empty())
                    break;

                string frameFilename = Path.Combine(frameDir, $"frame_{frameCount:D6}.jpg");
                Cv2.ImWrite(frameFilename, frame);

                framesBuffer.Add(frame.Clone());
                if(framesBuffer.Count == framesPerClip)
                {
                    CreateVideoClip(framesBuffer, clipsDir, clipCount, fps, width, height);
                    Console.WriteLine($"Created clip {clipCount} with frames {frameCount - framesPerClip + 1} to {frameCount}");

                    foreach(var bufferedFrame in framesBuffer)
                        bufferedFrame.Dispose();
                    
                    framesBuffer.Clear();
                    clipCount++;
                }

                frameCount++;
            }

            frame.Dispose();

            if(framesBuffer.Any())
            {
                CreateVideoClip(framesBuffer, clipsDir, clipCount, fps, width, height);
                Console.WriteLine($"Create final clip {clipCount} with {framesBuffer.Count} frames");

                foreach(var bufferedFrame in framesBuffer)
                    bufferedFrame.Dispose();
            }

            Console.WriteLine($"[Extraction] Video processing complete. Clips saved in: {clipsDir}");
            Console.WriteLine("[Extraction] Calling LSTM Python script for testing...");


            string lstmOutput = PythonRunner.Run("lstm.py", _videoPath);
            Console.WriteLine("[Extraction] Python returned:");
            Console.WriteLine(lstmOutput);




            return new StageResult
            {
                OutputImage = null,
                Features = new Dictionary<string, double>()
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine("[Extraction] Error: " + ex.Message);
            return new StageResult
            {
                OutputImage = null,
                Features = new Dictionary<string, double>()
            };
        }
    }

    private void CreateVideoClip(List<Mat> frames, string outputDir, int clipNumber, double fps, int width, int height)
    {
        string clipFilename = Path.Combine(outputDir, $"clip_{clipNumber:D4}.avi");
        using var writer = new VideoWriter(clipFilename, FourCC.XVID, fps, new Size(width, height));
        foreach(var frame in frames)
            writer.Write(frame);
    }


}

}





    // extraction clip() path:

    // lstm.py ->return path:?
    // predict.py path-> result ... || YOU!!!!!!!!.
    // save output....16pcs?




    // images run() -> 16pcs 
    // 
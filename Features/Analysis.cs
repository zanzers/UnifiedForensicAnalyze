using System;
using OpenCvSharp;
using System.Drawing;              
using System.Drawing.Imaging;       
using OpenCvSharp.Extensions;       
using System.IO;                   
using System.Linq;

namespace UnifiedForensicsAnalyze.Features
{
    public class PRNUStage : IAnalysisStage
    {
        public string Name => "PRNU";

        private Mat? numeratorSum;
        private Mat? denominatorSum;
        private int imageCount = 0;

        public StageResult Process(Mat input)
        {
            // Step 1: Ensure input is 3-channel float
            Mat Iout = new Mat();
            Cv2.CvtColor(input, Iout, ColorConversionCodes.BGR2RGB);
            Iout.ConvertTo(Iout, MatType.CV_32FC3, 1.0 / 255.0);

            // Step 2: Denoise
            Mat Iin = new Mat();
            Cv2.FastNlMeansDenoisingColored(input, Iin, 10, 10, 7, 21);
            Iin.ConvertTo(Iin, MatType.CV_32FC3, 1.0 / 255.0);

            // Step 3: Residual (noise)
            Mat W = Iout - Iin;

            // Step 4: PRNU numerator and denominator
            Mat num = new Mat();
            Cv2.Multiply(W, Iin, num);

            Mat den = new Mat();
            Cv2.Multiply(Iin, Iin, den);

            // Step 5: Initialize accumulators (3-channel float)
            if (numeratorSum == null)
            {
                numeratorSum = Mat.Zeros(input.Size(), MatType.CV_32FC3);
                denominatorSum = Mat.Zeros(input.Size(), MatType.CV_32FC3);
            }

            // Step 6: Accumulate
            Cv2.Add(numeratorSum, num, numeratorSum);
            Cv2.Add(denominatorSum, den, denominatorSum);
            imageCount++;

            // Step 7: Prevent division by zero
            Mat eps = Mat.Ones(denominatorSum.Size(), denominatorSum.Type()) * 1e-8f;
            Mat denomSafe = denominatorSum + eps;

            // Step 8: Divide
            Mat F = new Mat();
            Cv2.Divide(numeratorSum, denomSafe, F);

            // Step 9: Zero-mean normalize
            Scalar mean = Cv2.Mean(F);
            Mat FzeroMean = new Mat();
            Cv2.Subtract(F, new Scalar(mean.Val0, mean.Val1, mean.Val2), FzeroMean);

            // Step 10: Normalize magnitude
            Mat sq = new Mat();
            Cv2.Pow(FzeroMean, 2.0, sq);
            double norm = Math.Sqrt(Cv2.Sum(sq).Val0) + 1e-8;
            Mat FNormalized = FzeroMean / norm;

            // Step 11: Visualize
            Mat FVis = new Mat();
            Cv2.Normalize(FNormalized, FVis, 0, 255, NormTypes.MinMax);
            FVis.ConvertTo(FVis, MatType.CV_8UC1);


            Mat featureVector = FNormalized.Reshape(1, 1);
            Console.WriteLine($"Feature vector size: {featureVector.Cols} values");

            // Extract as float array
            featureVector.GetArray(out float[] data);

            Console.WriteLine($"Feature vector size: {data.Length}");
            Console.WriteLine("Sample (first 10 values):");
            for (int i = 0; i < Math.Min(10, data.Length); i++)
                Console.Write($"{data[i]:F6} ");
            Console.WriteLine();

            // ---- Statistical summary ----
            Scalar meanOut, stddev;
            Cv2.MeanStdDev(featureVector, out meanOut, out stddev);
            Console.WriteLine($"Mean: {meanOut.Val0:F6}, StdDev: {stddev.Val0:F6}");

            double energy = Cv2.Norm(featureVector, NormTypes.L2);
            Console.WriteLine($"Energy (L2 norm): {energy:F6}");


            return new StageResult
            {
                OutputImage = FVis,

            };
        }
    }


    public class ELAStage : IAnalysisStage
    {
        public string Name => "ELA";

        public StageResult Process(Mat input)
        {
            // Convert input to float 0..1
            Mat inputFloat = new Mat();
            input.ConvertTo(inputFloat, MatType.CV_32FC3, 1.0 / 255.0);

            // Encode to JPEG in memory
            Cv2.ImEncode(".jpg", input, out byte[] jpegBytes, new int[] { (int)ImwriteFlags.JpegQuality, 95 });

            // Decode back
            Mat jpegImg = Cv2.ImDecode(jpegBytes, ImreadModes.Color);
            jpegImg.ConvertTo(jpegImg, MatType.CV_32FC3, 1.0 / 255.0);

            // Compute absolute difference
            Mat ela = new Mat();
            Cv2.Absdiff(inputFloat, jpegImg, ela);

            // Feature vector (flatten float matrix)
            Mat featureVector = ela.Reshape(1, 1); // 1 row, all channels
            featureVector.GetArray(out float[] data);

            Console.WriteLine($"ELA Feature length: {data.Length}");
            // optional: print first 10 values for sanity check
            Console.WriteLine("Sample: " + string.Join(", ", data.Take(10).Select(v => v.ToString("F6"))));



            // Visualization for display
            Mat elaVis = new Mat();
            Cv2.Normalize(ela, elaVis, 0, 255, NormTypes.MinMax);
            elaVis.ConvertTo(elaVis, MatType.CV_8UC3);


            return new StageResult
            {
                OutputImage = elaVis,

            };
        }
    }

    public class SVDStage : IAnalysisStage
    {
        public string Name => "SVD";

        public StageResult Process(Mat input)
        {
            // 1) Convert image to grayscale
            Mat gray = new Mat();
            Cv2.CvtColor(input, gray, ColorConversionCodes.BGR2GRAY);
            gray.ConvertTo(gray, MatType.CV_32F);

            // 2) Perform Singular Value Decomposition
            Mat W = new();
            Mat U = new();
            Mat Vt = new();
            Cv2.SVDecomp(gray, W, U, Vt);

            // 3) Extract singular values
            int svCount = W.Rows * W.Cols;
            float[] svArray = new float[svCount];
            W.GetArray(out svArray);

            // 4) Convert to positive doubles and normalize
            double eps = 1e-12;
            double[] svDouble = svArray.Select(v => Math.Abs((double)v)).ToArray();
            double sumSv = svDouble.Sum() + eps;
            double[] svNorm = svDouble.Select(v => v / sumSv).ToArray();

            // 5) Compute feature statistics (for ML use)
            double top1 = svNorm.Length > 0 ? svNorm[0] : 0;
            double top5 = svNorm.Take(Math.Min(5, svNorm.Length)).Sum();
            double top10 = svNorm.Take(Math.Min(10, svNorm.Length)).Sum();
            double entropy = -svNorm.Sum(p => p > 0 ? p * Math.Log(p + eps) : 0);

            // 6) Print features for debugging / observation
            Console.WriteLine("---- SVD Features ----");
            Console.WriteLine($"Total Singular Values: {svCount}");
            Console.WriteLine($"Top 1: {top1:F6}");
            Console.WriteLine($"Top 5 Sum: {top5:F6}");
            Console.WriteLine($"Top 10 Sum: {top10:F6}");
            Console.WriteLine($"Entropy: {entropy:F6}");
            Console.WriteLine("----------------------");

            // 7) Return as feature vector (no image visualization)
            Mat wVis = new Mat(W.Rows, W.Cols, MatType.CV_8UC1);
            double minVal, maxVal;
            Cv2.MinMaxLoc(W, out minVal, out maxVal);
            Cv2.Normalize(W, wVis, 0, 255, NormTypes.MinMax);
            wVis.ConvertTo(wVis, MatType.CV_8UC1);

            return new StageResult
            {
                OutputImage = wVis
            };

            //             A high entropy and lower top1/top5 ratios → more likely tampered or AI.

            //             A low entropy and high top1/top5 ratios → more likely original or natural.
        }

    }


     public class IWTStage : IAnalysisStage
    {
        public string Name => "IWT";

        public StageResult Process(Mat input)
        {
            // Convert to grayscale
            Mat gray = new Mat();
            Cv2.CvtColor(input, gray, ColorConversionCodes.BGR2GRAY);
            gray.ConvertTo(gray, MatType.CV_32F);

            int rows = gray.Rows;
            int cols = gray.Cols;

            // Ensure even dimensions
            int rowsEven = rows - (rows % 2);
            int colsEven = cols - (cols % 2);
            Mat resized = gray.SubMat(0, rowsEven, 0, colsEven);

            // Split into even/odd rows & columns (Haar Integer Wavelet)
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

                    // Integer Haar transform
                    LL.Set<float>(y / 2, x / 2, (a + b + c + d) / 4f);
                    LH.Set<float>(y / 2, x / 2, (a + b - c - d) / 4f);
                    HL.Set<float>(y / 2, x / 2, (a - b + c - d) / 4f);
                    HH.Set<float>(y / 2, x / 2, (a - b - c + d) / 4f);
                }
            }

            // Compute statistical features
            double[] llFeatures = ComputeStats(LL);
            double[] lhFeatures = ComputeStats(LH);
            double[] hlFeatures = ComputeStats(HL);
            double[] hhFeatures = ComputeStats(HH);

            Console.WriteLine("IWT Features:");
            Console.WriteLine($"LL -> Mean: {llFeatures[0]:F6}, Var: {llFeatures[1]:F6}, Entropy: {llFeatures[2]:F6}");
            Console.WriteLine($"LH -> Mean: {lhFeatures[0]:F6}, Var: {lhFeatures[1]:F6}, Entropy: {lhFeatures[2]:F6}");
            Console.WriteLine($"HL -> Mean: {hlFeatures[0]:F6}, Var: {hlFeatures[1]:F6}, Entropy: {hlFeatures[2]:F6}");
            Console.WriteLine($"HH -> Mean: {hhFeatures[0]:F6}, Var: {hhFeatures[1]:F6}, Entropy: {hhFeatures[2]:F6}");

            // Visualization (combine subbands)
           Mat top = new Mat();
            Mat bottom = new Mat();
            Mat vis = new Mat();

            Cv2.HConcat(new Mat[] { LL, LH }, top);
            Cv2.HConcat(new Mat[] { HL, HH }, bottom);
            Cv2.VConcat(new Mat[] { top, bottom }, vis);

            // Normalize for view
            Cv2.Normalize(vis, vis, 0, 255, NormTypes.MinMax);
            vis.ConvertTo(vis, MatType.CV_8UC1);

            return new StageResult
            {
                OutputImage = vis
            };
        }

        private double[] ComputeStats(Mat m)
        {
            Scalar mean, stddev;
            Cv2.MeanStdDev(m, out mean, out stddev);

            // Flatten to array
            m.GetArray(out float[] arr);
            double meanVal = mean.Val0;
            double varVal = stddev.Val0 * stddev.Val0;

            // Compute entropy
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
        // “If LH_var and HH_entropy are above X → likely tampered.”
    }

}

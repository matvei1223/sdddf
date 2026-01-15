using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace LiveDetect;

public static class Program
{
    private sealed record Options(
        string ModelPath,
        string? NamesPath,
        float Confidence,
        float Iou,
        int ImageSize
    );

    private sealed record Detection(Rect Rect, float Confidence, int ClassId);

    public static void Main(string[] args)
    {
        var options = ParseArgs(args);
        var classNames = LoadNames(options.NamesPath);

        using var session = new InferenceSession(options.ModelPath);
        var inputName = session.InputMetadata.Keys.First();

        using var window = new Window("Live Detect");

        while (true)
        {
            using var frame = CapturePrimaryScreen();
            var resized = frame.Resize(new OpenCvSharp.Size(options.ImageSize, options.ImageSize));

            var inputTensor = BuildInputTensor(resized);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using var results = session.Run(inputs);
            var output = results.First().AsTensor<float>();

            var detections = ParseDetections(
                output,
                options.Confidence,
                options.Iou,
                options.ImageSize,
                frame.Width,
                frame.Height
            );

            foreach (var det in detections)
            {
                Cv2.Rectangle(frame, det.Rect, Scalar.Red, 2);
                var label = classNames.Count > det.ClassId
                    ? $"{classNames[det.ClassId]} {det.Confidence:0.00}"
                    : $"cls {det.ClassId} {det.Confidence:0.00}";
                Cv2.PutText(
                    frame,
                    label,
                    new Point(det.Rect.X, Math.Max(det.Rect.Y - 6, 0)),
                    HersheyFonts.HersheySimplex,
                    0.5,
                    Scalar.Red,
                    2
                );
            }

            window.ShowImage(frame);

            var key = Cv2.WaitKey(1);
            if (key == 'q' || key == 27)
            {
                break;
            }
        }
    }

    private static Options ParseArgs(string[] args)
    {
        string? modelPath = null;
        string? namesPath = null;
        float confidence = 0.35f;
        float iou = 0.5f;
        int imageSize = 640;

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            switch (arg)
            {
                case "--model":
                    modelPath = args[++i];
                    break;
                case "--names":
                    namesPath = args[++i];
                    break;
                case "--conf":
                    confidence = float.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--iou":
                    iou = float.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
                case "--img":
                    imageSize = int.Parse(args[++i], CultureInfo.InvariantCulture);
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("--model is required");
        }

        return new Options(modelPath, namesPath, confidence, iou, imageSize);
    }

    private static List<string> LoadNames(string? path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return new List<string>();
        }

        return File.ReadAllLines(path)
            .Select(line => line.Trim())
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .ToList();
    }

    private static Mat CapturePrimaryScreen()
    {
        var bounds = System.Windows.Forms.Screen.PrimaryScreen.Bounds;
        using var bitmap = new Bitmap(bounds.Width, bounds.Height);
        using var graphics = Graphics.FromImage(bitmap);
        graphics.CopyFromScreen(bounds.Left, bounds.Top, 0, 0, bitmap.Size);
        return BitmapConverter.ToMat(bitmap);
    }

    private static DenseTensor<float> BuildInputTensor(Mat image)
    {
        var chw = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });
        var index = 0;
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var color = image.At<Vec3b>(y, x);
                chw.Buffer.Span[index++] = color.Item2 / 255f;
            }
        }
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var color = image.At<Vec3b>(y, x);
                chw.Buffer.Span[index++] = color.Item1 / 255f;
            }
        }
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var color = image.At<Vec3b>(y, x);
                chw.Buffer.Span[index++] = color.Item0 / 255f;
            }
        }
        return chw;
    }

    private static List<Detection> ParseDetections(
        Tensor<float> output,
        float confThreshold,
        float iouThreshold,
        int inputSize,
        int originalWidth,
        int originalHeight)
    {
        var dims = output.Dimensions.ToArray();
        var channelsFirst = dims.Length == 3 && dims[1] < dims[2];
        var channels = channelsFirst ? dims[1] : dims[2];
        var boxes = channelsFirst ? dims[2] : dims[1];
        var classCount = channels - 4;

        var detections = new List<Detection>();
        for (var i = 0; i < boxes; i++)
        {
            float cx = GetValue(output, channelsFirst, 0, i, channels);
            float cy = GetValue(output, channelsFirst, 1, i, channels);
            float w = GetValue(output, channelsFirst, 2, i, channels);
            float h = GetValue(output, channelsFirst, 3, i, channels);

            var bestClass = -1;
            var bestScore = 0f;
            for (var c = 0; c < classCount; c++)
            {
                var score = GetValue(output, channelsFirst, 4 + c, i, channels);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestScore < confThreshold)
            {
                continue;
            }

            var x = (cx - w / 2) * originalWidth / inputSize;
            var y = (cy - h / 2) * originalHeight / inputSize;
            var width = w * originalWidth / inputSize;
            var height = h * originalHeight / inputSize;

            var rect = new Rect(
                (int)Math.Max(x, 0),
                (int)Math.Max(y, 0),
                (int)Math.Min(width, originalWidth - x),
                (int)Math.Min(height, originalHeight - y)
            );
            detections.Add(new Detection(rect, bestScore, bestClass));
        }

        return ApplyNms(detections, iouThreshold);
    }

    private static float GetValue(Tensor<float> output, bool channelsFirst, int channel, int index, int channels)
    {
        if (channelsFirst)
        {
            return output[0, channel, index];
        }

        return output[0, index, channel];
    }

    private static List<Detection> ApplyNms(List<Detection> detections, float iouThreshold)
    {
        var ordered = detections.OrderByDescending(d => d.Confidence).ToList();
        var results = new List<Detection>();

        while (ordered.Count > 0)
        {
            var best = ordered[0];
            ordered.RemoveAt(0);
            results.Add(best);

            for (var i = ordered.Count - 1; i >= 0; i--)
            {
                if (IoU(best.Rect, ordered[i].Rect) > iouThreshold)
                {
                    ordered.RemoveAt(i);
                }
            }
        }

        return results;
    }

    private static float IoU(Rect a, Rect b)
    {
        var intersection = a & b;
        if (intersection.Width <= 0 || intersection.Height <= 0)
        {
            return 0f;
        }

        var intersectionArea = intersection.Width * intersection.Height;
        var unionArea = a.Width * a.Height + b.Width * b.Height - intersectionArea;
        return unionArea == 0 ? 0f : intersectionArea / unionArea;
    }
}

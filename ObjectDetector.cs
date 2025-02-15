using OpenCvSharp.Dnn;
using OpenCvSharp;
using System.IO;

namespace WpfObjectDetection
{
    public class ObjectDetector
    {
		public ObjectDetector(String model_dir)
		{
			String cfgFile = "";
			String modelFile = "";
			if (Directory.Exists(model_dir)) {
				string[] fileEntries = Directory.GetFiles(model_dir);
				foreach (string fileName in fileEntries) {
					if (Path.GetExtension(fileName) == ".cfg")
						cfgFile = fileName;
					else if (Path.GetExtension(fileName) == ".weights")
						modelFile = fileName;
					else if (Path.GetExtension(fileName) == ".txt")
						classNames = File.ReadAllLines(fileName);
				}
			}

			if (String.IsNullOrEmpty(cfgFile) || String.IsNullOrEmpty(modelFile) || classNames.Length <= 0)
				throw new FileNotFoundException();

			net = Net.ReadNetFromDarknet(cfgFile, modelFile);
		}

		public int Detect(Mat image)
		{
			Clear();

			Mat inputBlob = CvDnn.BlobFromImage(image, 1/255f, new Size(416, 416), crop: false);
			net!.SetInput(inputBlob);
			var outBlobNames = net.GetUnconnectedOutLayersNames();
			var outputBlobs = outBlobNames.Select(toMat => new Mat()).ToArray();
			if (outBlobNames == null || outBlobNames.Length <= 0)
				return 0;

			net.Forward(outputBlobs, outBlobNames!);
			foreach (Mat prob in outputBlobs) {
				for (int p = 0; p < prob.Rows; p++) {
					float confidence = prob.At<float>(p, 4);
					if (confidence > 0.9) {
						Cv2.MinMaxLoc(prob.Row(p).ColRange(5, prob.Cols), out _, out _, out _, out Point classNumber);

						int classes = classNumber.X;
						float probability = prob.At<float>(p, classes + 5);

						if (probability > 0.9) {
							float centerX = prob.At<float>(p, 0) * image.Width;
							float centerY = prob.At<float>(p, 1) * image.Height;
							float width = prob.At<float>(p, 2) * image.Width;
							float height = prob.At<float>(p, 3) * image.Height;

							labels.Add(classNames[classes]);
							scores.Add(probability);
							bboxes.Add(new Rect((int)centerX - (int)width / 2, (int)centerY - (int)height / 2, (int)width, (int)height));
						}
					}
				}
			}
			return bboxes.Count;
		}

		public void Clear()
		{
			labels.Clear();
			scores.Clear();
			bboxes.Clear();
		}

		public List<string> Labels => labels;
		public List<float> Scores => scores;
		public List<Rect> Bboxes => bboxes;

		private readonly string[] classNames = [];

		private List<string> labels = [];
		private List<float> scores = [];
		private List<Rect> bboxes = [];

		private readonly Net? net;
	}
}

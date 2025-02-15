using MaterialDesignThemes.Wpf;
using Microsoft.Win32;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media.Imaging;
using System.Drawing;
using System.IO;

using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenCvSharp.Extensions;

using Window = System.Windows.Window;
using System.Diagnostics;
using System.Drawing.Imaging;

namespace WpfObjectDetection;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

		WindowStartupLocation = WindowStartupLocation.CenterScreen;
		var baseDir = AppDomain.CurrentDomain.BaseDirectory;
		var darknet_dir = System.IO.Path.Combine(baseDir, "darknet_model");
		if (!System.IO.Directory.Exists(darknet_dir)) {
#if DEBUG
			darknet_dir = darknet_dir.Replace("\\Debug\\", "\\Release\\");
#else
			darknet_dir = darknet_dir.Replace("\\Release\\", "\\Debug\\");
#endif
		}
		yoloDetector = new ObjectDetector(darknet_dir);
    }

	protected override void OnSourceInitialized(EventArgs e)
	{
		base.OnSourceInitialized(e);

		var isLightTheme = IsLightTheme();
		var source = (HwndSource)PresentationSource.FromVisual(this);
		ToggleBaseColour(source.Handle, !isLightTheme);

		// Detect when the theme changed
		source.AddHook((IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled) => {
			const int WM_SETTINGCHANGE = 0x001A;
			if (msg == WM_SETTINGCHANGE) {
				if (wParam == IntPtr.Zero && Marshal.PtrToStringUni(lParam) == "ImmersiveColorSet") {
					var isLightTheme = IsLightTheme();
					ToggleBaseColour(hwnd, !isLightTheme);
				}
			}

			return IntPtr.Zero;
		});
	}

	private readonly ObjectDetector yoloDetector;

	private Mat? ImageOrigin { get; set; }

	private Mat? ImageMatrix { get; set; }

	private String? ImageFilePath { get; set; }

	private void OnLoadImage(object sender, RoutedEventArgs e)
	{
		var dlg = new OpenFileDialog {
			Title = "Select Image",
			Filter = ""
		};
		var codecs = ImageCodecInfo.GetImageEncoders();
		var sep = string.Empty;
		var bmpFilter = string.Empty;
		foreach (var c in codecs) {
			string codecName = c.CodecName!.Substring(8).Replace("Codec", "Files").Trim();
			if (codecName.Contains("BMP"))
				bmpFilter = String.Format("|{0} ({1})|{1}", codecName, c.FilenameExtension);
			else {
				dlg.Filter = String.Format("{0}{1}{2} ({3})|{3}", dlg.Filter, sep, codecName, c.FilenameExtension);
				sep = "|";
			}
		}
		if (!String.IsNullOrEmpty(bmpFilter))
			dlg.Filter += bmpFilter;
		dlg.Filter = String.Format("{0}{1}{2} ({3})|{3}", dlg.Filter, sep, "All Files", "*.*");
		if (dlg.ShowDialog() == true) {
			ImageFilePath = dlg.FileName;
			ImagePathTextBox.Text = PathShortener(ImageFilePath, 48);
			ImageOrigin = Cv2.ImRead(ImageFilePath);
			yoloDetector.Clear();
			OnFilterChecked(this, e);
		}
	}

	private Mat? ProcessFilter(Mat? srcImage)
	{
		if (srcImage == null)
			return null;
		var image = srcImage.Clone();
		bool isFiltered = false;
		if ((bool)FilterBlur.IsChecked!) {
			Cv2.Blur(image, image, new OpenCvSharp.Size(5, 5));
			isFiltered = true;
		}
		if ((bool)FilterGrayscale.IsChecked!) {
			Cv2.CvtColor(image, image, ColorConversionCodes.BGR2GRAY);
			isFiltered = true;
		}
		if ((bool)FilterCanny.IsChecked!) {
			var canny = new Mat();
			Cv2.Canny(image, canny, 100, 200, 3, true);
			return canny;
		}
		return isFiltered ? image : srcImage;
	}

	private void OnFilterChecked(object sender, RoutedEventArgs e)
	{
		if (ImageOrigin != null) {
			if (yoloDetector.Bboxes.Count > 0) {
				var image = ImageOrigin.Clone();
				var bitmap = BitmapConverter.ToBitmap(ProcessFilter(image)!);
				var temp_bitmap = new Bitmap(bitmap.Width, bitmap.Height);
				using var g = Graphics.FromImage(temp_bitmap);
				{
					g.DrawImage(bitmap, 0, 0);
					g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
					var pen = new Pen(Color.Red, 2);
					var brush = new SolidBrush(Color.FromArgb(255 * 50 / 100, Color.Red));
					var str_brush = new SolidBrush(Color.White);
					var font = new Font("맑은 고딕", 11);
					CvDnn.NMSBoxes(yoloDetector.Bboxes, yoloDetector.Scores, 0.9f, 0.5f, out int[] indices);
					foreach (int i in indices) {
						var rc = yoloDetector.Bboxes[i];
						g.DrawRectangle(pen, rc.Left, rc.Top, rc.Right - rc.Left, rc.Bottom - rc.Top);
						var str_size = g.MeasureString(yoloDetector.Labels[i], font);
						g.FillRectangle(brush, rc.Left, rc.Top, str_size.Width, str_size.Height);
						g.DrawString(yoloDetector.Labels[i], font, str_brush, rc.Left + 1, rc.Top - 1);
					}
				}
				var converted = Convert(temp_bitmap);
				imgViewport.Source = converted;
			} else {
				ImageMatrix = ProcessFilter(ImageOrigin);
				if (ImageMatrix != null) {
					var converted = Convert(BitmapConverter.ToBitmap(ImageMatrix));
					imgViewport.Source = converted;
				}
			}
		}
	}

	private BitmapImage Convert(Bitmap src)
	{
		MemoryStream ms = new();
		((System.Drawing.Bitmap)src).Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
		BitmapImage image = new();
		image.BeginInit();
		ms.Seek(0, SeekOrigin.Begin);
		image.StreamSource = ms;
		image.EndInit();
		return image;
	}

	private void OnObjectDetection(object sender, RoutedEventArgs e)
	{
		if (ImageOrigin == null)
			return;
		var image = ImageOrigin.Clone();
		var _time = Stopwatch.StartNew();
		var count = yoloDetector.Detect(image);
		var ts = _time.Elapsed;
		ElapsedTextBlock.Text = $"Elapsed: {ts.TotalMilliseconds.ToString("0.00")}ms";
		if (count > 0) {
			OnFilterChecked(this, e);
		}
	}

	private readonly PaletteHelper _paletteHelper = new PaletteHelper();

	private void ToggleBaseColour(nint hwnd, bool isDark)
	{
		var theme = _paletteHelper.GetTheme();
		var baseTheme = isDark ? BaseTheme.Dark : BaseTheme.Light;
		theme.SetBaseTheme(baseTheme);
		_paletteHelper.SetTheme(theme);
		UseImmersiveDarkMode(hwnd, isDark);
	}

	private static bool IsLightTheme()
	{
		using var key = Registry.CurrentUser.OpenSubKey(@"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
		var value = key?.GetValue("AppsUseLightTheme");
		return value is int i && i > 0;
	}

	[DllImport("dwmapi.dll")]
	private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);

	private const int DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19;
	private const int DWMWA_USE_IMMERSIVE_DARK_MODE = 20;

	private static bool UseImmersiveDarkMode(IntPtr handle, bool enabled)
	{
		if (OperatingSystem.IsWindowsVersionAtLeast(10, 0, 17763)) {
			var attribute = DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1;
			if (OperatingSystem.IsWindowsVersionAtLeast(10, 0, 18985)) {
				attribute = DWMWA_USE_IMMERSIVE_DARK_MODE;
			}

			int useImmersiveDarkMode = enabled ? 1 : 0;
			return DwmSetWindowAttribute(handle, attribute, ref useImmersiveDarkMode, sizeof(int)) == 0;
		}

		return false;
	}

	[DllImport("shlwapi.dll", CharSet = CharSet.Auto)]
	static extern bool PathCompactPathEx([Out] StringBuilder pszOut, string szPath, int cchMax, int dwFlags);

	static string PathShortener(string path, int length)
	{
		StringBuilder sb = new StringBuilder(length + 1);
		PathCompactPathEx(sb, path, length, 0);
		return sb.ToString();
	}

	private void RadioButton_Checked(object sender, RoutedEventArgs e)
	{

	}
}

from gi.repository import Gst
import gi
gi.require_version('Gst', '1.0')

Gst.init(None)
print("GStreamer initialized successfully!")

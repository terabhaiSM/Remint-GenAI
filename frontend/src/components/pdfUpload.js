import React, { useState } from "react";
import axios from "axios";
import { Link, useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";

const PdfUpload = () => {
  const [title, setTitle] = useState("");
  const [pdfFile, setPdfFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const navigate = useNavigate();

  const handleTitleChange = (e) => {
    setTitle(e.target.value);
  };

  const handlePdfChange = (e) => {
    setPdfFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("title", title);
    formData.append("pdf", pdfFile);

    const response = await axios.post("http://127.0.0.1:5000/pdf", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    setVideoUrl(response.data.video_url);
    navigate("/video-preview", { replace: true });
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center">Get Video from PDF</h1>
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="form-group">
          <input
            type="text"
            className="form-control mb-3"
            name="title"
            placeholder="Enter title"
            required
            value={title}
            onChange={handleTitleChange}
          />
        </div>
        <div className="form-group">
          <input
            type="file"
            className="form-control-file mb-3"
            name="pdf"
            accept=".pdf"
            onChange={handlePdfChange}
          />
        </div>
        <button type="submit" className="btn btn-primary mb-3">
          Generate Video
        </button>
      </form>
      {/* Div to show video preview */}
      {videoUrl && (
        <div id="video-preview" className="video-preview mt-4">
          <video width="100%" controls>
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      )}
      <p>
        <Link to="/">Back to Video Generator</Link>
      </p>
    </div>
  );
};

export default PdfUpload;
import React, { useState } from "react";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link } from "react-router-dom";

const VideoGenerator = (props) => {
  const [topic, setTopic] = useState("");
  const [description, setDescription] = useState("");
  const [descriptionUrl, setDescriptionUrl] = useState("");
  const [flavor, setFlavor] = useState("Meme");
  const [videoUrl, setVideoUrl] = useState("");
  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("topic", topic);
    formData.append("flavor", flavor);
    formData.append("description", description);
    formData.append("description_url", descriptionUrl);
  
    const response = await axios.post("http://127.0.0.1:5000/", formData, {
      headers: {
        "Content-Type": "multipart/form-data", // Set the appropriate Content-Type header
      },
    });
    setVideoUrl(response.data.video_url);
    // props.history.push('/video-preview');
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center">Mental Health Video Generator</h1>
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="form-group">
          <input
            type="text"
            className="form-control mb-3"
            name="topic"
            placeholder="Enter a mental health topic"
            required
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>
        <div className="form-group">
          <textarea
            className="form-control mb-3"
            name="description"
            rows="3"
            placeholder="Enter description (Optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          ></textarea>
        </div>
        <div className="form-group">
          <input
            type="url"
            className="form-control mb-3"
            name="description_url"
            placeholder="Enter description URL (Optional)"
            value={descriptionUrl}
            onChange={(e) => setDescriptionUrl(e.target.value)}
          />
        </div>
        <div className="form-group">
          <select
            name="flavor"
            className="form-control mb-3"
            value={flavor}
            onChange={(e) => setFlavor(e.target.value)}
          >
            <option value="Meme">Meme</option>
            <option value="Advanced Insights">Advanced Insights</option>
            <option value="Warning">Warning</option>
            <option value="Informational">Informational</option>
            <option value="Psychoeducation">Psychoeducation</option>
            <option value="Shock Value">Shock Value</option>
            <option value="Virality">Virality</option>
            <option value="Negative News">Negative News</option>
            <option value="Positive News">Positive News</option>
          </select>
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
        <Link to="/pdfUpload">Generate Video from PDF</Link>
      </p>
    </div>
  );
};

export default VideoGenerator;

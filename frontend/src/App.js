import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import VideoGenerator from './components/videoGenerator';
import PdfUpload from './components/pdfUpload';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<VideoGenerator />} />
        <Route path="/pdfUpload" element={<PdfUpload />} />
      </Routes>
    </Router>
  );
};

export default App;
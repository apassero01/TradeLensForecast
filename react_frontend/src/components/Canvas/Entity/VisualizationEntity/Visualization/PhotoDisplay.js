import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaTrash, FaExpand, FaCompress, FaInfoCircle, FaSync } from 'react-icons/fa';

const PhotoDisplay = ({ visualization, sendStrategyRequest, entityId }) => {
  const [localImage, setLocalImage] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [imageInfo, setImageInfo] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Prioritize localImage over visualization.data
  const imageData = localImage !== null ? localImage : visualization?.data;
  const title = visualization?.config?.title || 'Image Display';

  useEffect(() => {
    if (imageData) {
      setErrorMessage(null);
    } else {
      setImageInfo(null);
    }
  }, [imageData]);

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    accept: { 'image/*': [] },
    noClick: true,
    noKeyboard: true,
    maxFiles: 1,
    onDrop: handleFileUpload,
    onDropRejected: () => {
      setErrorMessage('Please select a valid image file');
    }
  });

  async function handleFileUpload(acceptedFiles) {
    if (!acceptedFiles || acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const dataUrl = await readFileAsDataURL(file);
      console.log('dataUrl generated:', dataUrl); // Debug log

      setImageInfo({
        format: file.type.split('/')[1]?.toUpperCase() || 'Unknown',
        size: formatFileSize(file.size),
        name: file.name
      });

      setLocalImage(dataUrl);

      if (visualization.onDataChange) {
        visualization.onDataChange(dataUrl);
      }

      if (sendStrategyRequest) {
        await sendStrategyRequest({
          strategy_name: 'SetAttributesStrategy',
          param_config: {
            attribute_map: {
              'imagePath': dataUrl,
              'imageFileName': file.name,
              'imageType': file.type,
              'imageLastModified': file.lastModified
            }
          }
        });
      }
    } catch (error) {
      console.error('Error handling file upload:', error);
      setErrorMessage(`Error uploading file: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${Math.round(bytes / (1024 * 1024))} MB`;
  }

  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = (e) => reject(e);
      reader.readAsDataURL(file);
    });
  }

  async function handleRemoveImage() {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      if (visualization.onDataChange) {
        visualization.onDataChange(null);
      }

      if (sendStrategyRequest) {
        await sendStrategyRequest({
          strategy_name: 'SetAttributesStrategy',
          param_config: {
            attribute_map: {
              'imagePath': null,
              'imageFileName': null,
              'imageType': null,
              'imageLastModified': null
            }
          }
        });
      }

      setImageInfo(null);
      setLocalImage(null);
    } catch (error) {
      console.error('Error removing image:', error);
      setErrorMessage('Error removing image');
    } finally {
      setIsLoading(false);
    }
  }

  function toggleFullscreen() {
    setIsFullscreen(!isFullscreen);
  }

  const LoadingSpinner = () => (
    <div className="text-gray-400 flex flex-col items-center">
      <FaSync size={48} className="mb-4 animate-spin" />
      <p>Processing image...</p>
    </div>
  );

  const FallbackImageDisplay = () => (
    <div className="w-full h-full flex items-center justify-center">
      <img 
        src={imageData} 
        alt="Uploaded content"
        className={`max-w-full max-h-full object-contain ${isFullscreen ? 'w-auto h-auto' : ''}`}
        onError={() => {
          setErrorMessage('Image format not supported by your browser');
        }}
      />
    </div>
  );

  return (
    <div className="flex flex-col w-full h-full nodrag bg-gray-800">
      <div className="flex-none border-b border-gray-700 p-2 flex justify-between items-center">
        <h2 className="text-gray-200 text-lg">{title}</h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={open}
            disabled={isLoading}
            className="px-2 py-1 bg-blue-600 rounded hover:bg-blue-700 flex items-center disabled:opacity-50"
            title="Upload image"
          >
            {isLoading ? <FaSync className="mr-1 animate-spin" /> : <FaUpload className="mr-1" />} Upload
          </button>
          {imageData && (
            <>
              <button
                onClick={handleRemoveImage}
                disabled={isLoading}
                className="px-2 py-1 bg-red-600 rounded hover:bg-red-700 flex items-center disabled:opacity-50"
                title="Remove image"
              >
                <FaTrash className="mr-1" /> Remove
              </button>
              <button
                onClick={toggleFullscreen}
                className="px-2 py-1 bg-gray-700 rounded hover:bg-gray-600"
                title={isFullscreen ? "Exit fullscreen" : "View fullscreen"}
              >
                {isFullscreen ? <FaCompress /> : <FaExpand />}
              </button>
            </>
          )}
        </div>
      </div>

      {errorMessage && (
        <div className="flex-none bg-red-700 px-3 py-2 text-sm text-white">
          <strong>Error:</strong> {errorMessage}
        </div>
      )}

      {imageInfo && (
        <div className="flex-none bg-gray-700 px-3 py-1 text-xs text-gray-300 flex items-center">
          <FaInfoCircle className="mr-2" />
          <span>Format: <strong>{imageInfo.format}</strong></span>
          <span className="mx-3">|</span>
          <span>Size: <strong>{imageInfo.size}</strong></span>
          {imageInfo.name && (
            <>
              <span className="mx-3">|</span>
              <span>File: <strong>{imageInfo.name}</strong></span>
            </>
          )}
        </div>
      )}

      <div 
        {...getRootProps()} 
        className={`flex-grow overflow-auto flex items-center justify-center p-4 ${isFullscreen ? 'fixed inset-0 z-50 bg-black' : ''} ${isDragActive ? 'bg-blue-900 bg-opacity-20' : ''}`}
      >
        <input {...getInputProps()} />
        {isLoading ? (
          <LoadingSpinner />
        ) : imageData ? (
          <FallbackImageDisplay />
        ) : (
          <div className="text-gray-400 flex flex-col items-center">
            <FaUpload size={48} className="mb-4" />
            <p>{isDragActive ? 'Drop the image here' : 'No image uploaded'}</p>
            <p className="text-xs mt-1 text-gray-500">Drag & drop an image here, or click the Upload button</p>
            <p className="text-xs mt-1 text-gray-500">Supports most image formats including PNG, JPG, GIF, WEBP</p>
            <button 
              onClick={open}
              disabled={isLoading}
              className="mt-4 px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              Select Image
            </button>
          </div>
        )}
      </div>

      {isFullscreen && imageData && (
        <button
          onClick={toggleFullscreen}
          className="fixed top-4 right-4 z-50 bg-black bg-opacity-50 rounded-full p-2 text-white"
          title="Exit fullscreen"
        >
          <FaCompress size={20} />
        </button>
      )}
    </div>
  );
};

export default PhotoDisplay;
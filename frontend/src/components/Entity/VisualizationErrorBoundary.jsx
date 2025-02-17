import React, { Component } from 'react';

class VisualizationErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render shows the fallback UI.
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error, or send it to an analytics service
    console.error("Error rendering visualization:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '1rem', backgroundColor: '#333', color: 'red' }}>
          There was an error rendering this visualization.
        </div>
      );
    }
    return this.props.children;
  }
}

export default VisualizationErrorBoundary; 
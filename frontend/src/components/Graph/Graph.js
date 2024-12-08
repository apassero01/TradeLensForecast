import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';

cytoscape.use(dagre);

const transformData = (data) => {
  const elements = [];
  
  // Add nodes
  data.nodes.forEach(node => {
    elements.push({
      data: { ...node, id: node.id.toString() }
    });
  });
  
  // Add edges
  data.edges.forEach(edge => {
    elements.push({
      data: {
        id: `${edge.source}-${edge.target}`,
        source: edge.source.toString(),
        target: edge.target.toString()
      }
    });
  });
  
  return elements;
};

const Graph = ({ data, onNodeClick }) => {
  const graphRef = useRef(null);
  const nodePositions = useRef(new Map());
  const cyRef = useRef(null);

  useEffect(() => {
    if (!graphRef.current) return;

    const cy = cytoscape({
      container: graphRef.current,
      elements: transformData(data),
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#4B5563',
            'label': 'data(label)',
            'color': '#E5E7EB',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '12px',
            'width': '40px',
            'height': '40px',
            'border-width': '2px',
            'border-color': '#374151'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#4B5563',
            'target-arrow-color': '#4B5563',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier'
          }
        },
        {
          selector: ':selected',
          style: {
            'background-color': '#3B82F6',
            'border-color': '#60A5FA',
            'line-color': '#60A5FA',
            'target-arrow-color': '#60A5FA'
          }
        }
      ],
      layout: {
        name: 'dagre',
        rankDir: 'TB',
        nodeDimensionsIncludeLabels: true,
        spacingFactor: 1.5,
        rankSep: 100,
        animate: true,
        animationDuration: 500,
        positions: (node) => nodePositions.current.get(node.id())
      },
      // Disable all keyboard events and node removal
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: true,
      selectionType: 'single',
      autoungrabify: false,
      autounselectify: false,
      keyboard: {
        enabled: false,
      },
      removeWithFields: false // Prevent node removal
    });

    cyRef.current = cy;

    // Prevent node removal
    cy.on('remove', 'node', (event) => {
      event.preventDefault();
      return false;
    });

    // Save positions after drag or layout
    const savePositions = () => {
      cy.nodes().forEach(node => {
        nodePositions.current.set(node.id(), node.position());
      });
    };

    cy.on('dragfree', savePositions);
    cy.on('layoutstop', savePositions);
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      onNodeClick(node.data());
    });

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [data, onNodeClick]);

  // Prevent default keyboard events at the window level
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault();
        e.stopPropagation();
        // Unselect any selected nodes when backspace is pressed
        if (cyRef.current) {
          cyRef.current.$(':selected').unselect();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, []);

  return <div ref={graphRef} style={{ width: '100%', height: '100%' }} />;
};

export default Graph; 
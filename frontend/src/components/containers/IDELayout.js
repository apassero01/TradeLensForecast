import React from 'react';
import { ResizableBox } from 'react-resizable';
import 'react-resizable/css/styles.css';

function DynamicIDELayout({ panels }) {
    // Generate gridTemplateAreas dynamically based on panel positions
    const gridTemplateAreas = panels
        .map((panel) => `"${panel.gridArea}"`)
        .join(' ');

    // Extract unique row and column configuration based on panel layout
    const uniqueRows = [...new Set(panels.map((panel) => panel.row))];
    const uniqueCols = [...new Set(panels.map((panel) => panel.col))];

    return (
        <div
            className="h-full w-full p-4 grid gap-4"
            style={{
                gridTemplateAreas: gridTemplateAreas,  // Define grid areas based on panels
                gridTemplateRows: `repeat(${uniqueRows.length}, 1fr)`,
                gridTemplateColumns: `repeat(${uniqueCols.length}, 1fr)`,
            }}
        >
            {panels.map((panel, index) => (
                <ResizableBox
                    key={index}
                    className="border border-gray-300 bg-white rounded-lg shadow-lg flex flex-col"
                    width={panel.initialWidth || 300}
                    height={panel.initialHeight || 200}
                    minConstraints={[150, 150]}
                    maxConstraints={[800, 800]}
                    style={{ gridArea: panel.gridArea }}
                >
                    <div className="h-full w-full p-4">{panel.component}</div>
                </ResizableBox>
            ))}
        </div>
    );
}

export default DynamicIDELayout;
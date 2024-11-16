import React from 'react';

function GridRow({ components, columnWidths }) {
    return (
        <div
            className="grid gap-4 w-full"
            style={{
                gridTemplateColumns: columnWidths
                    .map((width) => `${width}%`)
                    .join(' '), // Set column widths as percentages
            }}
        >
            {components.map((componentData, index) => (
                <div
                    key={index}
                    className="border border-gray-300 flex h-full w-full"
                    style={{
                        gridColumn: componentData.colSpan
                            ? `span ${componentData.colSpan}`
                            : 'auto',
                    }}
                >
                    <div className="flex-grow h-full w-full flex justify-center items-center">
                        {/* This div ensures the child fills all available space */}
                        {componentData.component || null}
                    </div>
                </div>
            ))}
        </div>
    );
}

export default GridRow;
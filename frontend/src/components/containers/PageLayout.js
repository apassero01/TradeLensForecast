import React from 'react';
import GridRow from './GridRow';

function PageLayout({ layout }) {
    return (
        <div className="h-full w-full p-4 space-y-4">
            {layout.map((row, rowIndex) => (
                <GridRow
                    key={rowIndex}
                    components={row.components}
                    columnWidths={row.columnWidths} // Pass column widths as percentages
                />
            ))}
        </div>
    );
}

export default PageLayout;
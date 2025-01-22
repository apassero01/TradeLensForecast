// EntityMenu.js
import React from 'react';
import { ControlledMenu, MenuItem } from '@szhsin/react-menu';
import '@szhsin/react-menu/dist/index.css';

export function EntityMenuTrigger({ children, onCopyId, onNewStrategy }) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [anchorPoint, setAnchorPoint] = React.useState({ x: 0, y: 0 });

  const handleContextMenu = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setAnchorPoint({ x: e.clientX, y: e.clientY });
    setIsOpen(true);
  };

  return (
    <>
      <div 
        className="w-full h-full"
        onContextMenu={handleContextMenu}
      >
        {children}
      </div>
      <ControlledMenu
        state={isOpen ? 'open' : 'closed'}
        anchorPoint={anchorPoint}
        onClose={() => setIsOpen(false)}
        menuClassName="bg-gray-900 text-gray-200 border border-gray-700 rounded shadow-lg py-1 min-w-[160px]"
        itemClassName="px-4 py-2 hover:bg-gray-800 hover:text-gray-200 cursor-pointer"
        portal={true}
      >
        <MenuItem onClick={onCopyId}>Copy Entity ID</MenuItem>
        <MenuItem onClick={onNewStrategy}>New Strategy</MenuItem>
      </ControlledMenu>
    </>
  );
}

// We don't need the separate EntityMenu component anymore
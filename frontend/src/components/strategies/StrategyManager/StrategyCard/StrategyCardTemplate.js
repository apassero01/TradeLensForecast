// StrategyCardTemplate.js
import React, { useState } from 'react';
import PropTypes from 'prop-types';
import JSONEditorModal from '../../utils/JSONEditorModal';

function StrategyCardTemplate({
  strategy,
  isSubmitted,
  onSubmit,
  onRemove,
  renderOpenEditorTrigger,
  ...props
}) {
  const [isEditing, setIsEditing] = useState(false);

  const handleOpenEditor = () => {
    setIsEditing(true);
  };

  const handleSaveConfig = (newConfig) => {
    // Update the strategy's config
    strategy.config = newConfig;
    setIsEditing(false);
  };

  return (
    <div className="strategy-card" {...props}>
      {/* Custom Content */}
      {props.children}

      {/* Open Editor Trigger */}
      {renderOpenEditorTrigger ? (
        renderOpenEditorTrigger(handleOpenEditor)
      ) : (
        <button onClick={handleOpenEditor}>Edit Config</button>
      )}

      {/* JSON Editor Modal */}
      {isEditing && (
        <JSONEditorModal
          initialConfig={strategy.config}
          onSave={handleSaveConfig}
          onCancel={() => setIsEditing(false)}
        />
      )}
    </div>
  );
}

StrategyCardTemplate.propTypes = {
  strategy: PropTypes.object.isRequired,
  isSubmitted: PropTypes.bool.isRequired,
  onSubmit: PropTypes.func,
  onRemove: PropTypes.func,
  renderOpenEditorTrigger: PropTypes.func,
  children: PropTypes.node,
};

export default StrategyCardTemplate;
// StrategyContainerTemplate.js
import React from 'react';
import PropTypes from 'prop-types';

function StrategyContainerTemplate({
  children,
  ...props
}) {
  return (
    <div className="strategy-container">
      {children}
    </div>
  );
}

StrategyContainerTemplate.propTypes = {
  children: PropTypes.node,
};

export default StrategyContainerTemplate;
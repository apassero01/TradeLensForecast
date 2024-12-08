// StrategyManager/StrategyManager.js
import React from 'react';
import PropTypes from 'prop-types';

function StrategyManager({
  StrategyCardComponent,
  StrategyContainerComponent,
  useStrategyDataHook,
  strategyConfig,
  showSelectionMenu = true,
  onSubmit,
  ...otherProps
}) {
  const strategyData = useStrategyDataHook(strategyConfig);

  const {
    availableStrategies,
    tempStrategies,
    existingStrategies,
    error,
    handleAddStrategy,
    handleRemoveTempStrategy,
    handleSubmit,
  } = strategyData;

  const submitHandler = onSubmit || handleSubmit;

  return (
    <StrategyContainerComponent
      availableStrategies={availableStrategies}
      tempStrategies={tempStrategies}
      existingStrategies={existingStrategies}
      error={error}
      handleAddStrategy={handleAddStrategy}
      handleRemoveTempStrategy={handleRemoveTempStrategy}
      handleSubmit={submitHandler}
      StrategyCardComponent={StrategyCardComponent}
      showSelectionMenu={showSelectionMenu}
      {...otherProps}
    />
  );
}

StrategyManager.propTypes = {
  StrategyCardComponent: PropTypes.elementType.isRequired,
  StrategyContainerComponent: PropTypes.elementType.isRequired,
  useStrategyDataHook: PropTypes.func.isRequired,
  strategyConfig: PropTypes.object.isRequired,
  showSelectionMenu: PropTypes.bool,
  onSubmit: PropTypes.func,
  // Other props
};

export default StrategyManager;
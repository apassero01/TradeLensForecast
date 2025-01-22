import { createPortal } from 'react-dom';

const ModalRoot = ({ children }) => {
  const modalRoot = document.getElementById('modal-root');
  if (!modalRoot) return null;
  
  return createPortal(children, modalRoot);
};

export default ModalRoot; 
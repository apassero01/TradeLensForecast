import { atom } from 'recoil';

export const notificationAtom = atom({
  key: 'notificationAtom',
  default: null, // { message: string, type: 'error' | 'success' | 'info' }
}); 
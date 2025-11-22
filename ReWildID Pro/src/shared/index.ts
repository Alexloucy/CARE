// Mock shared exports
export const selectCurrentUser = (state: any) => state.user.currentUser;
export const selectUnreadCount = () => 0;
export const selectTotalUnreadCount = () => 0;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const patchUserPreferencesThunk = (_args: any) => async (_dispatch: any) => { };

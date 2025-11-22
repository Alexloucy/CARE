import { configureStore, createSlice } from '@reduxjs/toolkit'
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux'

// Mock user slice
const userSlice = createSlice({
    name: 'user',
    initialState: {
        currentUser: {
            preferences: {
                sidebarVisibility: {}
            }
        }
    },
    reducers: {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        updatePreferences: (state, _action) => {
            return state;
        }
    }
})

export const store = configureStore({
    reducer: {
        user: userSlice.reducer
    }
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export const useAppDispatch: () => AppDispatch = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector

import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds
  headers: {
    'Content-Type': 'application/json',
  },
})

export const uploadAudio = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  
  const response = await apiClient.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  
  return response.data
}

export const getClasses = async () => {
  const response = await apiClient.get('/classes')
  return response.data
}

export const healthCheck = async () => {
  const response = await apiClient.get('/')
  return response.data
}

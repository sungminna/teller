import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

// Configure notifications
export const initializeNotifications = async () => {
  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('news-updates', {
      name: 'News Updates',
      importance: Notifications.AndroidImportance.DEFAULT,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#1976D2',
    });

    await Notifications.setNotificationChannelAsync('processing', {
      name: 'News Processing',
      importance: Notifications.AndroidImportance.HIGH,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#FF9800',
    });
  }

  // Request permissions
  const { status: existingStatus } = await Notifications.getPermissionsAsync();
  let finalStatus = existingStatus;

  if (existingStatus !== 'granted') {
    const { status } = await Notifications.requestPermissionsAsync();
    finalStatus = status;
  }

  if (finalStatus !== 'granted') {
    console.warn('Notification permissions not granted');
    return null;
  }

  // Get push token
  if (Device.isDevice) {
    try {
      const token = await Notifications.getExpoPushTokenAsync({
        projectId: Constants.expoConfig?.extra?.eas?.projectId,
      });
      console.log('Push token:', token.data);
      return token.data;
    } catch (error) {
      console.error('Failed to get push token:', error);
      return null;
    }
  } else {
    console.warn('Must use physical device for push notifications');
    return null;
  }
};

// Send local notification
export const sendLocalNotification = async (
  title: string,
  body: string,
  data?: any,
  channelId: string = 'news-updates'
) => {
  try {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: true,
      },
      trigger: null, // Send immediately
      identifier: `local-${Date.now()}`,
    });
  } catch (error) {
    console.error('Failed to send local notification:', error);
  }
};

// Handle notification received
export const handleNotificationReceived = (notification: Notifications.Notification) => {
  console.log('Notification received:', notification);
  
  // Handle different notification types
  const { data } = notification.request.content;
  
  if (data?.type === 'news_processed') {
    // News processing completed
    sendLocalNotification(
      '뉴스 처리 완료',
      '새로운 뉴스가 준비되었습니다!',
      data,
      'processing'
    );
  } else if (data?.type === 'news_update') {
    // New news available
    sendLocalNotification(
      '새 뉴스 업데이트',
      '관심있는 새 뉴스가 있습니다.',
      data
    );
  }
};

// Handle notification response (when user taps notification)
export const handleNotificationResponse = (response: Notifications.NotificationResponse) => {
  console.log('Notification response:', response);
  
  const { data } = response.notification.request.content;
  
  // Navigate based on notification type
  if (data?.newsId) {
    // Navigate to specific news item
    // This would be handled by the navigation system
    return {
      action: 'navigate',
      screen: 'Player',
      params: { newsId: data.newsId },
    };
  }
  
  return null;
};

// Schedule periodic notifications
export const schedulePeriodicNotification = async (
  title: string,
  body: string,
  intervalMinutes: number = 60
) => {
  try {
    await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        sound: true,
      },
      trigger: {
        seconds: intervalMinutes * 60,
        repeats: true,
      },
      identifier: `periodic-${Date.now()}`,
    });
  } catch (error) {
    console.error('Failed to schedule periodic notification:', error);
  }
};

// Cancel all notifications
export const cancelAllNotifications = async () => {
  try {
    await Notifications.cancelAllScheduledNotificationsAsync();
  } catch (error) {
    console.error('Failed to cancel notifications:', error);
  }
};

// Register for push notifications
export const registerForPushNotifications = async (userId: string) => {
  const token = await initializeNotifications();
  
  if (token) {
    try {
      // Send token to backend
      const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/users/push-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`, // Use auth token
        },
        body: JSON.stringify({
          userId,
          pushToken: token,
          platform: Platform.OS,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to register push token');
      }

      console.log('Push token registered successfully');
    } catch (error) {
      console.error('Failed to register push token:', error);
    }
  }
}; 
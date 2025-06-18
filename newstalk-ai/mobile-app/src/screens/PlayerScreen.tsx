import React, { useEffect, useState, useCallback } from 'react';
import {
  View,
  StyleSheet,
  Dimensions,
  Alert,
  Share,
  ScrollView,
  TextInput,
} from 'react-native';
import {
  Card,
  Text,
  IconButton,
  Button,
  Chip,
  Surface,
  useTheme,
  ActivityIndicator,
  ProgressBar,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRoute, useNavigation, RouteProp } from '@react-navigation/native';
import TrackPlayer, {
  State,
  usePlaybackState,
  useProgress,
  useTrackPlayerEvents,
  Event,
} from 'react-native-track-player';
import Slider from '@react-native-community/slider';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { format } from 'date-fns';
import { ko } from 'date-fns/locale';

// Store
import { useNewsStore } from '../store/newsStore';
import { useAuthStore } from '../store/authStore';

// Services
import { setupPlayerService } from '../services/audioService';

// Types
import { RootStackParamList } from '../../App';

type PlayerScreenRouteProp = RouteProp<RootStackParamList, 'Player'>;

const { width, height } = Dimensions.get('window');

const PLAYBACK_SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

export default function PlayerScreen() {
  const theme = useTheme();
  const route = useRoute<PlayerScreenRouteProp>();
  const navigation = useNavigation();
  
  const { newsId } = route.params;
  
  // Store state
  const { news, toggleBookmark, submitFeedback } = useNewsStore();
  const { user } = useAuthStore();
  
  // Player state
  const playbackState = usePlaybackState();
  const progress = useProgress();
  
  // Local state
  const [currentNews, setCurrentNews] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [showSpeedModal, setShowSpeedModal] = useState(false);
  const [rating, setRating] = useState(0);
  const [showRatingModal, setShowRatingModal] = useState(false);
  
  // 대화형 Q&A 상태 추가
  const [showQAInterface, setShowQAInterface] = useState(false);
  const [qaMessages, setQaMessages] = useState<Array<{id: string, type: 'user' | 'assistant', content: string, timestamp: Date}>>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isProcessingQuestion, setIsProcessingQuestion] = useState(false);
  const [recommendedQuestions, setRecommendedQuestions] = useState<string[]>([]);
  const [conversationContext, setConversationContext] = useState<string[]>([]);

  // Find current news item
  useEffect(() => {
    const newsItem = news.find(item => item.id === newsId);
    if (newsItem) {
      setCurrentNews(newsItem);
      if (newsItem.audioUrl) {
        setupAudio(newsItem);
      }
    }
  }, [newsId, news]);

  // Setup audio player
  const setupAudio = useCallback(async (newsItem) => {
    try {
      setIsLoading(true);
      
      await setupPlayerService();
      
      // Add track to player
      await TrackPlayer.add({
        id: newsItem.id,
        url: newsItem.audioUrl,
        title: newsItem.title,
        artist: 'NewsTeam AI',
        duration: newsItem.duration,
        artwork: undefined, // Could add news thumbnail
      });
      
      setIsLoading(false);
    } catch (error) {
      console.error('Audio setup error:', error);
      setIsLoading(false);
      Alert.alert('오류', '오디오를 로드할 수 없습니다.');
    }
  }, []);

  // Handle play/pause
  const handlePlayPause = useCallback(async () => {
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      
      if (playbackState === State.Playing) {
        await TrackPlayer.pause();
      } else {
        await TrackPlayer.play();
      }
    } catch (error) {
      console.error('Play/pause error:', error);
    }
  }, [playbackState]);

  // Handle seek
  const handleSeek = useCallback(async (value: number) => {
    try {
      await TrackPlayer.seekTo(value);
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    } catch (error) {
      console.error('Seek error:', error);
    }
  }, []);

  // Handle skip forward/backward
  const handleSkip = useCallback(async (seconds: number) => {
    try {
      const currentPosition = progress.position;
      const newPosition = Math.max(0, Math.min(progress.duration, currentPosition + seconds));
      await TrackPlayer.seekTo(newPosition);
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    } catch (error) {
      console.error('Skip error:', error);
    }
  }, [progress]);

  // Handle speed change
  const handleSpeedChange = useCallback(async (speed: number) => {
    try {
      await TrackPlayer.setRate(speed);
      setPlaybackSpeed(speed);
      setShowSpeedModal(false);
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    } catch (error) {
      console.error('Speed change error:', error);
    }
  }, []);

  // Handle share
  const handleShare = useCallback(async () => {
    if (!currentNews) return;
    
    try {
      await Share.share({
        message: `${currentNews.title}\n\n${currentNews.summary}\n\nNewsTeam AI로 들어보세요!`,
        title: currentNews.title,
      });
    } catch (error) {
      console.error('Share error:', error);
    }
  }, [currentNews]);

  // Handle bookmark toggle
  const handleBookmarkToggle = useCallback(() => {
    if (!currentNews) return;
    
    toggleBookmark(currentNews.id);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  }, [currentNews, toggleBookmark]);

  // Handle rating submission
  const handleRatingSubmit = useCallback(async () => {
    if (!currentNews || rating === 0) return;
    
    try {
      await submitFeedback(currentNews.id, rating);
      setShowRatingModal(false);
      setRating(0);
      Alert.alert('감사합니다', '피드백이 제출되었습니다.');
    } catch (error) {
      Alert.alert('오류', '피드백 제출에 실패했습니다.');
    }
  }, [currentNews, rating, submitFeedback]);

  // Format time
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Track player events
  useTrackPlayerEvents([Event.PlaybackError], (event) => {
    if (event.type === Event.PlaybackError) {
      console.error('Playback error:', event);
      Alert.alert('재생 오류', '오디오 재생 중 오류가 발생했습니다.');
    }
  });

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      TrackPlayer.reset();
    };
  }, []);

  // 대화형 Q&A 기능 추가
  const handleAudioComplete = useCallback(async () => {
    // 오디오 재생 완료 시 자동으로 Q&A 모드 제안
    setTimeout(() => {
      setShowQAInterface(true);
      generateRecommendedQuestions();
    }, 2000);
  }, [currentNews]);

  const generateRecommendedQuestions = useCallback(async () => {
    if (!currentNews) return;
    
    try {
      // 뉴스 내용 기반 추천 질문 생성
      const questions = [
        `${currentNews.title}에 대해 더 자세히 알려주세요`,
        "이 뉴스와 관련된 배경 정보가 궁금해요",
        "비슷한 사례가 있었나요?",
        "이 문제의 향후 전망은 어떤가요?",
        "관련된 다른 뉴스도 있나요?"
      ];
      
      setRecommendedQuestions(questions);
    } catch (error) {
      console.error('추천 질문 생성 실패:', error);
    }
  }, [currentNews]);

  const handleQuestionSubmit = useCallback(async (question: string) => {
    if (!question.trim() || isProcessingQuestion) return;
    
    setIsProcessingQuestion(true);
    
    // 사용자 메시지 추가
    const userMessage = {
      id: `user_${Date.now()}`,
      type: 'user' as const,
      content: question,
      timestamp: new Date()
    };
    
    setQaMessages(prev => [...prev, userMessage]);
    setCurrentQuestion('');
    
    try {
      // AI 응답 요청
      const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/qa/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user?.accessToken}`,
        },
        body: JSON.stringify({
          question,
          article_id: currentNews?.id,
          context: conversationContext,
          user_preferences: user?.preferences
        }),
      });
      
      if (!response.ok) {
        throw new Error('AI 응답 요청 실패');
      }
      
      const aiResponse = await response.json();
      
      // AI 응답 메시지 추가
      const assistantMessage = {
        id: `assistant_${Date.now()}`,
        type: 'assistant' as const,
        content: aiResponse.answer,
        timestamp: new Date()
      };
      
      setQaMessages(prev => [...prev, assistantMessage]);
      
      // 대화 맥락 업데이트 (최대 5회 유지)
      setConversationContext(prev => {
        const newContext = [...prev, question, aiResponse.answer];
        return newContext.slice(-10); // 최대 5회 대화 유지
      });
      
    } catch (error) {
      console.error('질문 처리 실패:', error);
      Alert.alert('오류', '질문을 처리할 수 없습니다. 다시 시도해주세요.');
    } finally {
      setIsProcessingQuestion(false);
    }
  }, [currentNews, user, isProcessingQuestion, conversationContext]);

  const handleRecommendedQuestionPress = useCallback((question: string) => {
    setCurrentQuestion(question);
    handleQuestionSubmit(question);
  }, [handleQuestionSubmit]);

  if (!currentNews) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={theme.colors.primary} />
          <Text variant="bodyLarge" style={{ marginTop: 16 }}>
            뉴스를 로드하는 중...
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* News Info Card */}
        <Card style={styles.newsCard}>
          <Card.Content>
            <View style={styles.newsHeader}>
              <Chip style={styles.categoryChip}>{currentNews.category}</Chip>
              <View style={styles.newsActions}>
                <IconButton
                  icon={currentNews.isBookmarked ? 'bookmark' : 'bookmark-outline'}
                  iconColor={currentNews.isBookmarked ? theme.colors.primary : theme.colors.onSurface}
                  onPress={handleBookmarkToggle}
                />
                <IconButton
                  icon="share-variant"
                  iconColor={theme.colors.onSurface}
                  onPress={handleShare}
                />
              </View>
            </View>
            
            <Text variant="headlineSmall" style={styles.newsTitle}>
              {currentNews.title}
            </Text>
            
            <Text variant="bodyMedium" style={styles.newsSummary}>
              {currentNews.summary}
            </Text>
            
            {currentNews.quality && (
              <View style={styles.qualityContainer}>
                <Text variant="bodySmall" style={{ color: theme.colors.onSurfaceVariant }}>
                  품질 점수: {Math.round(currentNews.quality * 100)}%
                </Text>
              </View>
            )}
          </Card.Content>
        </Card>

        {/* Audio Player */}
        <Surface style={styles.playerContainer} elevation={2}>
          {isLoading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={theme.colors.primary} />
              <Text variant="bodyMedium" style={{ marginTop: 8 }}>
                오디오 로딩 중...
              </Text>
            </View>
          ) : (
            <>
              {/* Progress Bar */}
              <View style={styles.progressContainer}>
                <Text variant="bodySmall" style={styles.timeText}>
                  {formatTime(progress.position)}
                </Text>
                <Slider
                  style={styles.progressSlider}
                  minimumValue={0}
                  maximumValue={progress.duration}
                  value={progress.position}
                  onSlidingComplete={handleSeek}
                  minimumTrackTintColor={theme.colors.primary}
                  maximumTrackTintColor={theme.colors.outline}
                  thumbStyle={{ backgroundColor: theme.colors.primary }}
                />
                <Text variant="bodySmall" style={styles.timeText}>
                  {formatTime(progress.duration)}
                </Text>
              </View>

              {/* Player Controls */}
              <View style={styles.controlsContainer}>
                <IconButton
                  icon="rewind"
                  size={32}
                  iconColor={theme.colors.onSurface}
                  onPress={() => handleSkip(-15)}
                />
                
                <IconButton
                  icon={playbackState === State.Playing ? 'pause-circle' : 'play-circle'}
                  size={64}
                  iconColor={theme.colors.primary}
                  onPress={handlePlayPause}
                  style={styles.playButton}
                />
                
                <IconButton
                  icon="fast-forward"
                  size={32}
                  iconColor={theme.colors.onSurface}
                  onPress={() => handleSkip(15)}
                />
              </View>

              {/* Additional Controls */}
              <View style={styles.additionalControls}>
                <Button
                  mode="outlined"
                  onPress={() => setShowSpeedModal(true)}
                  style={styles.speedButton}
                >
                  {playbackSpeed}x
                </Button>
                
                <Button
                  mode="outlined"
                  onPress={() => setShowRatingModal(true)}
                  style={styles.ratingButton}
                >
                  평가하기
                </Button>
              </View>
            </>
          )}
        </Surface>
      </ScrollView>

      {/* Speed Selection Modal */}
      {showSpeedModal && (
        <View style={styles.modalOverlay}>
          <Surface style={styles.speedModal} elevation={8}>
            <Text variant="titleMedium" style={styles.modalTitle}>
              재생 속도 선택
            </Text>
            <View style={styles.speedOptions}>
              {PLAYBACK_SPEEDS.map((speed) => (
                <Button
                  key={speed}
                  mode={playbackSpeed === speed ? 'contained' : 'outlined'}
                  onPress={() => handleSpeedChange(speed)}
                  style={styles.speedOption}
                >
                  {speed}x
                </Button>
              ))}
            </View>
            <Button
              mode="text"
              onPress={() => setShowSpeedModal(false)}
              style={styles.modalCloseButton}
            >
              닫기
            </Button>
          </Surface>
        </View>
      )}

      {/* Rating Modal */}
      {showRatingModal && (
        <View style={styles.modalOverlay}>
          <Surface style={styles.ratingModal} elevation={8}>
            <Text variant="titleMedium" style={styles.modalTitle}>
              이 뉴스는 어떠셨나요?
            </Text>
            <View style={styles.ratingContainer}>
              {[1, 2, 3, 4, 5].map((star) => (
                <IconButton
                  key={star}
                  icon={star <= rating ? 'star' : 'star-outline'}
                  iconColor={star <= rating ? '#FFD700' : theme.colors.outline}
                  size={32}
                  onPress={() => setRating(star)}
                />
              ))}
            </View>
            <View style={styles.ratingActions}>
              <Button
                mode="text"
                onPress={() => setShowRatingModal(false)}
              >
                취소
              </Button>
              <Button
                mode="contained"
                onPress={handleRatingSubmit}
                disabled={rating === 0}
              >
                제출
              </Button>
            </View>
          </Surface>
        </View>
      )}

      {/* 대화형 Q&A 인터페이스 */}
      {showQAInterface && (
        <Card style={styles.qaCard}>
          <Card.Content>
            <View style={styles.qaHeader}>
              <Text variant="titleMedium" style={styles.qaTitle}>
                더 궁금한 것이 있나요?
              </Text>
              <IconButton
                icon="close"
                size={20}
                onPress={() => setShowQAInterface(false)}
              />
            </View>
            
            {/* 대화 히스토리 */}
            {qaMessages.length > 0 && (
              <ScrollView style={styles.qaHistory} showsVerticalScrollIndicator={false}>
                {qaMessages.map((message) => (
                  <View
                    key={message.id}
                    style={[
                      styles.qaMessage,
                      message.type === 'user' ? styles.userMessage : styles.assistantMessage,
                      { backgroundColor: message.type === 'user' ? theme.colors.primary + '20' : theme.colors.surface }
                    ]}
                  >
                    <Text variant="bodyMedium" style={styles.qaMessageText}>
                      {message.content}
                    </Text>
                    <Text variant="bodySmall" style={styles.qaTimestamp}>
                      {format(message.timestamp, 'HH:mm', { locale: ko })}
                    </Text>
                  </View>
                ))}
              </ScrollView>
            )}
            
            {/* 추천 질문 */}
            {qaMessages.length === 0 && recommendedQuestions.length > 0 && (
              <View style={styles.recommendedQuestions}>
                <Text variant="bodyMedium" style={styles.recommendedTitle}>
                  추천 질문:
                </Text>
                {recommendedQuestions.slice(0, 3).map((question, index) => (
                  <Chip
                    key={index}
                    onPress={() => handleRecommendedQuestionPress(question)}
                    style={styles.questionChip}
                    textStyle={styles.questionChipText}
                  >
                    {question}
                  </Chip>
                ))}
              </View>
            )}
            
            {/* 질문 입력 */}
            <View style={styles.qaInputContainer}>
              <TextInput
                mode="outlined"
                placeholder="궁금한 점을 물어보세요..."
                value={currentQuestion}
                onChangeText={setCurrentQuestion}
                multiline
                maxLength={200}
                style={styles.qaInput}
                right={
                  <TextInput.Icon
                    icon="send"
                    onPress={() => handleQuestionSubmit(currentQuestion)}
                    disabled={!currentQuestion.trim() || isProcessingQuestion}
                  />
                }
              />
              
              {isProcessingQuestion && (
                <View style={styles.qaProcessing}>
                  <ActivityIndicator size="small" color={theme.colors.primary} />
                  <Text variant="bodySmall" style={styles.qaProcessingText}>
                    답변을 생성하고 있습니다...
                  </Text>
                </View>
              )}
            </View>
          </Card.Content>
        </Card>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  newsCard: {
    margin: 16,
    marginBottom: 8,
  },
  newsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  categoryChip: {
    alignSelf: 'flex-start',
  },
  newsActions: {
    flexDirection: 'row',
  },
  newsTitle: {
    marginBottom: 8,
    lineHeight: 28,
  },
  newsSummary: {
    lineHeight: 20,
    opacity: 0.8,
  },
  qualityContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,0,0,0.1)',
  },
  playerContainer: {
    margin: 16,
    marginTop: 8,
    padding: 20,
    borderRadius: 16,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  timeText: {
    minWidth: 40,
    textAlign: 'center',
  },
  progressSlider: {
    flex: 1,
    height: 40,
    marginHorizontal: 8,
  },
  controlsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
  },
  playButton: {
    marginHorizontal: 20,
  },
  additionalControls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  speedButton: {
    minWidth: 80,
  },
  ratingButton: {
    minWidth: 100,
  },
  modalOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  speedModal: {
    margin: 20,
    padding: 20,
    borderRadius: 16,
    minWidth: 280,
  },
  ratingModal: {
    margin: 20,
    padding: 20,
    borderRadius: 16,
    minWidth: 300,
  },
  modalTitle: {
    textAlign: 'center',
    marginBottom: 16,
  },
  speedOptions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 16,
  },
  speedOption: {
    minWidth: 60,
  },
  ratingContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  ratingActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  modalCloseButton: {
    alignSelf: 'center',
  },
  // 대화형 Q&A 스타일 추가
  qaCard: {
    margin: 16,
    marginTop: 8,
  },
  qaHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  qaTitle: {
    fontWeight: 'bold',
  },
  qaHistory: {
    maxHeight: 200,
    marginBottom: 16,
  },
  qaMessage: {
    padding: 12,
    borderRadius: 12,
    marginBottom: 8,
  },
  userMessage: {
    alignSelf: 'flex-end',
    maxWidth: '80%',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    maxWidth: '80%',
  },
  qaMessageText: {
    marginBottom: 4,
  },
  qaTimestamp: {
    opacity: 0.6,
    textAlign: 'right',
  },
  recommendedQuestions: {
    marginBottom: 16,
  },
  recommendedTitle: {
    marginBottom: 8,
    fontWeight: '500',
  },
  questionChip: {
    marginBottom: 8,
    alignSelf: 'flex-start',
  },
  questionChipText: {
    fontSize: 12,
  },
  qaInputContainer: {
    position: 'relative',
  },
  qaInput: {
    minHeight: 56,
  },
  qaProcessing: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  qaProcessingText: {
    marginLeft: 8,
    opacity: 0.7,
  },
}); 
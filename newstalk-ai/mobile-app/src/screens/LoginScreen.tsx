import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import {
  Text,
  TextInput,
  Button,
  Card,
  useTheme,
  HelperText,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// Store
import { useAuthStore } from '../store/authStore';

// Validation schema
const loginSchema = z.object({
  email: z.string().email('올바른 이메일을 입력해주세요'),
  password: z.string().min(6, '비밀번호는 최소 6자 이상이어야 합니다'),
});

const registerSchema = z.object({
  name: z.string().min(2, '이름은 최소 2자 이상이어야 합니다'),
  email: z.string().email('올바른 이메일을 입력해주세요'),
  password: z.string().min(6, '비밀번호는 최소 6자 이상이어야 합니다'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: '비밀번호가 일치하지 않습니다',
  path: ['confirmPassword'],
});

type LoginForm = z.infer<typeof loginSchema>;
type RegisterForm = z.infer<typeof registerSchema>;

export default function LoginScreen() {
  const theme = useTheme();
  const { login, register, isLoading } = useAuthStore();
  
  const [isRegisterMode, setIsRegisterMode] = useState(false);

  // Login form
  const {
    control: loginControl,
    handleSubmit: handleLoginSubmit,
    formState: { errors: loginErrors },
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
  });

  // Register form
  const {
    control: registerControl,
    handleSubmit: handleRegisterSubmit,
    formState: { errors: registerErrors },
  } = useForm<RegisterForm>({
    resolver: zodResolver(registerSchema),
  });

  // Handle login
  const onLogin = async (data: LoginForm) => {
    try {
      await login(data.email, data.password);
    } catch (error) {
      Alert.alert('로그인 실패', '이메일 또는 비밀번호가 올바르지 않습니다.');
    }
  };

  // Handle register
  const onRegister = async (data: RegisterForm) => {
    try {
      await register(data.email, data.password, data.name);
    } catch (error) {
      Alert.alert('회원가입 실패', '회원가입 중 오류가 발생했습니다.');
    }
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <KeyboardAvoidingView
        style={styles.keyboardAvoid}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text variant="displaySmall" style={[styles.title, { color: theme.colors.primary }]}>
              NewsTeam AI
            </Text>
            <Text variant="bodyLarge" style={[styles.subtitle, { color: theme.colors.onSurfaceVariant }]}>
              {isRegisterMode ? '새 계정을 만들어보세요' : 'AI가 읽어주는 뉴스'}
            </Text>
          </View>

          {/* Form Card */}
          <Card style={styles.formCard}>
            <Card.Content style={styles.formContent}>
              {isRegisterMode ? (
                // Register Form
                <>
                  <Controller
                    control={registerControl}
                    name="name"
                    render={({ field: { onChange, onBlur, value } }) => (
                      <TextInput
                        label="이름"
                        mode="outlined"
                        onBlur={onBlur}
                        onChangeText={onChange}
                        value={value}
                        error={!!registerErrors.name}
                        style={styles.input}
                        autoCapitalize="words"
                      />
                    )}
                  />
                  <HelperText type="error" visible={!!registerErrors.name}>
                    {registerErrors.name?.message}
                  </HelperText>

                  <Controller
                    control={registerControl}
                    name="email"
                    render={({ field: { onChange, onBlur, value } }) => (
                      <TextInput
                        label="이메일"
                        mode="outlined"
                        onBlur={onBlur}
                        onChangeText={onChange}
                        value={value}
                        error={!!registerErrors.email}
                        style={styles.input}
                        keyboardType="email-address"
                        autoCapitalize="none"
                      />
                    )}
                  />
                  <HelperText type="error" visible={!!registerErrors.email}>
                    {registerErrors.email?.message}
                  </HelperText>

                  <Controller
                    control={registerControl}
                    name="password"
                    render={({ field: { onChange, onBlur, value } }) => (
                      <TextInput
                        label="비밀번호"
                        mode="outlined"
                        onBlur={onBlur}
                        onChangeText={onChange}
                        value={value}
                        error={!!registerErrors.password}
                        style={styles.input}
                        secureTextEntry
                      />
                    )}
                  />
                  <HelperText type="error" visible={!!registerErrors.password}>
                    {registerErrors.password?.message}
                  </HelperText>

                  <Controller
                    control={registerControl}
                    name="confirmPassword"
                    render={({ field: { onChange, onBlur, value } }) => (
                      <TextInput
                        label="비밀번호 확인"
                        mode="outlined"
                        onBlur={onBlur}
                        onChangeText={onChange}
                        value={value}
                        error={!!registerErrors.confirmPassword}
                        style={styles.input}
                        secureTextEntry
                      />
                    )}
                  />
                  <HelperText type="error" visible={!!registerErrors.confirmPassword}>
                    {registerErrors.confirmPassword?.message}
                  </HelperText>

                  <Button
                    mode="contained"
                    onPress={handleRegisterSubmit(onRegister)}
                    loading={isLoading}
                    disabled={isLoading}
                    style={styles.submitButton}
                  >
                    회원가입
                  </Button>
                </>
              ) : (
                // Login Form
                <>
                  <Controller
                    control={loginControl}
                    name="email"
                    render={({ field: { onChange, onBlur, value } }) => (
                      <TextInput
                        label="이메일"
                        mode="outlined"
                        onBlur={onBlur}
                        onChangeText={onChange}
                        value={value}
                        error={!!loginErrors.email}
                        style={styles.input}
                        keyboardType="email-address"
                        autoCapitalize="none"
                      />
                    )}
                  />
                  <HelperText type="error" visible={!!loginErrors.email}>
                    {loginErrors.email?.message}
                  </HelperText>

                  <Controller
                    control={loginControl}
                    name="password"
                    render={({ field: { onChange, onBlur, value } }) => (
                      <TextInput
                        label="비밀번호"
                        mode="outlined"
                        onBlur={onBlur}
                        onChangeText={onChange}
                        value={value}
                        error={!!loginErrors.password}
                        style={styles.input}
                        secureTextEntry
                      />
                    )}
                  />
                  <HelperText type="error" visible={!!loginErrors.password}>
                    {loginErrors.password?.message}
                  </HelperText>

                  <Button
                    mode="contained"
                    onPress={handleLoginSubmit(onLogin)}
                    loading={isLoading}
                    disabled={isLoading}
                    style={styles.submitButton}
                  >
                    로그인
                  </Button>
                </>
              )}

              {/* Toggle Mode */}
              <Button
                mode="text"
                onPress={() => setIsRegisterMode(!isRegisterMode)}
                style={styles.toggleButton}
              >
                {isRegisterMode ? '이미 계정이 있으신가요? 로그인' : '계정이 없으신가요? 회원가입'}
              </Button>
            </Card.Content>
          </Card>

          {/* Features */}
          <View style={styles.features}>
            <Text variant="titleMedium" style={[styles.featuresTitle, { color: theme.colors.onSurface }]}>
              NewsTeam AI 특징
            </Text>
            <View style={styles.featureItem}>
              <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
                • AI가 뉴스를 요약하고 음성으로 읽어드립니다
              </Text>
            </View>
            <View style={styles.featureItem}>
              <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
                • 개인 관심사에 맞춘 맞춤형 뉴스 제공
              </Text>
            </View>
            <View style={styles.featureItem}>
              <Text variant="bodyMedium" style={{ color: theme.colors.onSurfaceVariant }}>
                • 오프라인에서도 뉴스 청취 가능
              </Text>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardAvoid: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    padding: 20,
    justifyContent: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    textAlign: 'center',
  },
  formCard: {
    marginBottom: 24,
  },
  formContent: {
    padding: 20,
  },
  input: {
    marginBottom: 4,
  },
  submitButton: {
    marginTop: 16,
    marginBottom: 8,
  },
  toggleButton: {
    marginTop: 8,
  },
  features: {
    alignItems: 'center',
  },
  featuresTitle: {
    marginBottom: 16,
    fontWeight: '600',
  },
  featureItem: {
    marginBottom: 8,
    alignSelf: 'stretch',
  },
}); 
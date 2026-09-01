import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the app heading', () => {
  render(<App />);
  const heading = screen.getByRole('heading', { level: 3, name: /Predictive Maintenance Risk Assessment/i });
  expect(heading).toBeInTheDocument();
});

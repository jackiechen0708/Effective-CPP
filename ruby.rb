require 'date'

def two_sum(a, k)
  if a.length < 2
    return -1
  end
  result = nil
  (0..a.length - 1).each {|i|
    (i..a.length - 1).each {|j|
      sum = a.at(i) + a.at(j)
      if sum < k
        if result == nil
          result = sum
        else
          if result < sum
            result = sum
          end
        end
      end
    }
  }
  return (result == nil ? -1 : result)
end

def sorted_squares(a)
  (0..a.length - 1).each {|i|
    a[i] = a.at(i) * a.at(i)
  }
  return a.sort
end


def unique_total(a)
  freq = a.inject(Hash.new(0)) {|h, v| h[v] += 1; h}
  result = 0
  freq.each do |key, value|
    if value == 1
      result += key
    end
  end
  return result
end


def unique_char(s)
  freq = s.each_char.inject(Hash.new(0)) {|h, v| h[v] += 1; h}
  result = nil
  freq.each do |key, value|
    if value == 1
      tmp = s.index(key)
      if result == nil || tmp < result
        result = tmp
      end
    end
  end
  return (result == nil ? -1 : result)
end

def move_word(s)
  s = s.strip
  s = s.downcase
  s = s.each_char
  str1 = Array.new
  str2 = Array.new
  for i in s
    if i == 'a' || i == 'e' || i == 'i' || i == 'o' || i == 'u'
      str2.push(i)
    else
      str1.push(i)
    end
  end
  return (str1 + str2).join('')
end

def reverse_words(s)
  s = s.split
  words = Array.new
  for word in s
    words.push(move_word(word))
  end
  return words.reverse.join(' ')
end


class Movie
  def initialize(n, r)
    if n == nil
      raise ArgumentError
    end
    if r == nil || r == '' || !r.match(/\d{2}-\d{2}-\d{4}/)
      raise ArgumentError
    end
    @name, @release_date = n, r
  end


  def getName
    @name
  end

  def getReleaseDate
    @release_date
  end

  def setName=(value)
    @name = value
  end

  def setReleaseDate=(value)
    @release_date = value
  end

  def released_on
    begin
      date = Date.strptime(@release_date, '%m-%d-%Y')
      return @name +' - ' + date.strftime("%B %d %Y")
    rescue
      return "Invalid Date"
    end

  end

  def is_released
    begin
      date = Date.strptime(@release_date, '%m-%d-%Y')
      if date < Date.today
        return "true"
      else
        return "false"
      end
    rescue
      return "Invalid Date"
    end

  end
  
end
